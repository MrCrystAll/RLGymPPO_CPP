#include <RLGymPPO_CPP/PPO/PPOLearner.h>

#include <RLGymPPO_CPP/Util/TorchFuncs.h>

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>

using namespace torch;

Tensor _CopyParams(nn::Module* mod) {
	return torch::nn::utils::parameters_to_vector(mod->parameters()).cpu();
}

void _CopyModelParamsHalf(nn::Module* from, nn::Module* to) {
	RG_NOGRAD;
	try {
		auto fromParams = from->parameters();
		auto toParams = to->parameters();
		for (int i = 0; i < fromParams.size(); i++) {
			auto scaledParams = fromParams[i].to(RG_HALFPERC_TYPE);
			toParams[i].copy_(scaledParams, true);
		}
	} catch (std::exception& e) {
		RG_ERR_CLOSE("_CopyModelParamsHalf() exception: " << e.what());
	}
}

RLGPC::PPOLearner::PPOLearner(int obsSpaceSize, int actSpaceSize, PPOLearnerConfig _config, Device _device) 
	: config(_config), device(_device) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize must be a multiple of config.miniBatchSize");

	policy = new DiscretePolicy(obsSpaceSize, actSpaceSize, config.policyLayerSizes, device, config.policyTemperature);
	valueNet = new ValueEstimator(obsSpaceSize, config.criticLayerSizes, device);

	if (config.halfPrecModels) {
		policyHalf = new DiscretePolicy(obsSpaceSize, actSpaceSize, config.policyLayerSizes, device);
		valueNetHalf = new ValueEstimator(obsSpaceSize, config.criticLayerSizes, device);

		_CopyModelParamsHalf(policy, policyHalf);
		_CopyModelParamsHalf(valueNet, valueNetHalf);

		policyHalf->to(RG_HALFPERC_TYPE);
		valueNetHalf->to(RG_HALFPERC_TYPE);
	} else {
		policyHalf = NULL;
		valueNetHalf = NULL;
	}
	policyOptimizer = new optim::Adam(policy->parameters(), optim::AdamOptions(config.policyLR));
	valueOptimizer = new optim::Adam(valueNet->parameters(), optim::AdamOptions(config.criticLR));
	valueLossFn = nn::MSELoss();

	if (config.measureGradientNoise) {
		noiseTrackerPolicy = new GradNoiseTracker(config.batchSize, config.gradientNoiseUpdateInterval, config.gradientNoiseAvgDecay);
		noiseTrackerValueNet = new GradNoiseTracker(config.batchSize, config.gradientNoiseUpdateInterval, config.gradientNoiseAvgDecay);
	} else {
		noiseTrackerPolicy = NULL;
		noiseTrackerValueNet = NULL;
	}
}

// Structure to hold minibatch computation results
struct MinibatchResult {
	torch::Tensor policyLoss;
	torch::Tensor valueLoss;
	float entropy;
	float kl;
	float ratio;
	float clipFraction;
	bool hasPolicy;
	bool hasCritic;
};

void RLGPC::PPOLearner::Learn(ExperienceBuffer* expBuffer, Report& report) {

	int
		numIterations = 0,
		numMinibatchIterations = 0;
	float
		meanEntropy = 0,
		meanDivergence = 0,
		meanValLoss = 0,
		meanRatio = 0;
	FList clipFractions = {};

	// Save parameters first
	auto policyBefore = _CopyParams(policy);
	auto criticBefore = _CopyParams(valueNet);

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;

	Timer totalTimer = {};
	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		auto batches = expBuffer->GetAllBatchesShuffled(config.batchSize);

		for (auto& batch : batches) {
			auto batchActs = batch.actions;
			auto batchOldProbs = batch.logProbs;
			auto batchObs = batch.states;
			auto batchTargetValues = batch.values;
			auto batchAdvantages = batch.advantages;

			batchActs = batchActs.view({ config.batchSize, -1 });
			policyOptimizer->zero_grad();
			valueOptimizer->zero_grad();

			// Accumulate losses from all minibatches
			torch::Tensor totalPolicyLoss;
			torch::Tensor totalValueLoss;
			bool hasAnyPolicyLoss = false;
			bool hasAnyValueLoss = false;

			// Determine effective minibatch size - use consistent sizing for both CPU and GPU
			int effectiveMinibatchSize = config.miniBatchSize;
			if (this->device.is_cpu() && this->minibatchThreadPool) {
				// For multithreading, ensure we can evenly divide the work
				int numThreads = this->minibatchThreadPool->threads.size();
				effectiveMinibatchSize = config.batchSize / numThreads;
				effectiveMinibatchSize = RS_MAX(effectiveMinibatchSize, 1); // Ensure at least 1
			}

			// Process minibatches
			if (this->device.is_cpu() && this->minibatchThreadPool) {
				// Multithreaded processing for CPU
				if (!this->minibatchThreadPool) {
					int numThreads = std::thread::hardware_concurrency();
					numThreads += numThreads / 2; // Seems to be slightly faster
					this->minibatchThreadPool = new ThreadPool(numThreads);
				}

				// Thread-safe storage for results
				std::mutex resultsMutex;
				std::vector<MinibatchResult> results;

				auto fnRunMinibatch = [&](int start, int stop) {
					MinibatchResult result = {};
					result.hasPolicy = false;
					result.hasCritic = false;

					float batchSizeRatio = (stop - start) / (float)config.batchSize;

					// Send everything to the device and enforce correct shapes
					auto acts = batchActs.slice(0, start, stop).to(device, true, true);
					auto obs = batchObs.slice(0, start, stop).to(device, true, true);
					
					auto advantages = batchAdvantages.slice(0, start, stop).to(device, true, true);
					auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, true, true);
					auto targetValues = batchTargetValues.slice(0, start, stop).to(device, true, true);

					Timer timer = {};
					auto vals = valueNet->Forward(obs); // 11%
					std::lock_guard<std::mutex> lock(resultsMutex);
					report.Accum("PPO Value Estimate Time", timer.Elapsed());
					timer.Reset();

					torch::Tensor logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
					if (trainPolicy) {
						// Get policy log probs & entropy
						DiscretePolicy::BackpropResult bpResult = policy->GetBackpropData(obs, acts); // 13%

						logProbs = bpResult.actionLogProbs;
						entropy = bpResult.entropy;

						logProbs = logProbs.view_as(oldProbs);
						report.Accum("PPO Backprop Data Time", timer.Elapsed());

						// Compute PPO loss
						ratio = exp(logProbs - oldProbs);
						result.ratio = ratio.mean().detach().cpu().item<float>();

						clipped = clamp(
							ratio, 1 - config.clipRange, 1 + config.clipRange
						);

						// Compute policy loss
						policyLoss = -min(
							ratio * advantages, clipped * advantages
						).mean();
						ppoLoss = (policyLoss - entropy * config.entCoef) * batchSizeRatio;
						
						result.policyLoss = ppoLoss;
						result.hasPolicy = true;
						result.entropy = entropy.cpu().detach().item<float>();

						// Compute KL divergence & clip fraction using SB3 method for reporting
						{
							RG_NOGRAD;
							auto logRatio = logProbs - oldProbs;
							auto klTensor = (exp(logRatio) - 1) - logRatio;
							result.kl = klTensor.mean().detach().cpu().item<float>();
							result.clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat)).cpu().item<float>();
						}
					}

					if (trainCritic) {
						// Compute value loss
						vals = vals.view_as(targetValues);
						torch::Tensor valueLoss = valueLossFn(vals, targetValues) * batchSizeRatio;
						result.valueLoss = valueLoss;
						result.hasCritic = true;
					}

					// Store results thread-safely
					{
						std::lock_guard<std::mutex> lock(resultsMutex);
						results.push_back(result);
						report.Accum("PPO Gradient Time", timer.Elapsed());
						numMinibatchIterations += 1;
					}
				};

				// Launch threads
				for (int mbs = 0; mbs < config.batchSize; mbs += effectiveMinibatchSize) {
					int start = mbs;
					int stop = RS_MIN(start + effectiveMinibatchSize, config.batchSize);
					this->minibatchThreadPool->StartJob(std::bind(fnRunMinibatch, start, stop));
				}

				// Wait for all threads to complete
				while (this->minibatchThreadPool->GetNumRunningJobs() > 0)
					RG_SLEEP(1);

				// Accumulate results from all threads
				for (const auto& result : results) {
					if (result.hasPolicy) {
						if (!hasAnyPolicyLoss) {
							totalPolicyLoss = result.policyLoss;
							hasAnyPolicyLoss = true;
						} else {
							totalPolicyLoss += result.policyLoss;
						}
						meanEntropy += result.entropy;
						meanDivergence += result.kl;
						meanRatio += result.ratio;
						clipFractions.push_back(result.clipFraction);
					}
					if (result.hasCritic) {
						if (!hasAnyValueLoss) {
							totalValueLoss = result.valueLoss;
							hasAnyValueLoss = true;
						} else {
							totalValueLoss += result.valueLoss;
						}
						meanValLoss += result.valueLoss.cpu().detach().item<float>();
					}
				}

			} else {
				// Single-threaded processing for GPU or when threading is disabled
				for (int mbs = 0; mbs < config.batchSize; mbs += effectiveMinibatchSize) {
					int start = mbs;
					int stop = RS_MIN(start + effectiveMinibatchSize, config.batchSize);

					float batchSizeRatio = (stop - start) / (float)config.batchSize;

					// Send everything to the device and enforce correct shapes
					auto acts = batchActs.slice(0, start, stop).to(device, true, true);
					auto obs = batchObs.slice(0, start, stop).to(device, true, true);
					
					auto advantages = batchAdvantages.slice(0, start, stop).to(device, true, true);
					auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, true, true);
					auto targetValues = batchTargetValues.slice(0, start, stop).to(device, true, true);

					Timer timer = {};
					auto vals = valueNet->Forward(obs);
					report.Accum("PPO Value Estimate Time", timer.Elapsed());

					timer.Reset();
					torch::Tensor logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
					if (trainPolicy) {
						DiscretePolicy::BackpropResult bpResult = policy->GetBackpropData(obs, acts);

						logProbs = bpResult.actionLogProbs;
						entropy = bpResult.entropy;

						logProbs = logProbs.view_as(oldProbs);
						report.Accum("PPO Backprop Data Time", timer.Elapsed());

						// Compute PPO loss
						ratio = exp(logProbs - oldProbs);
						meanRatio += ratio.mean().detach().cpu().item<float>();
						clipped = clamp(
							ratio, 1 - config.clipRange, 1 + config.clipRange
						);

						// Compute policy loss
						policyLoss = -min(
							ratio * advantages, clipped * advantages
						).mean();
						ppoLoss = (policyLoss - entropy * config.entCoef) * batchSizeRatio;

						// Accumulate policy loss
						if (!hasAnyPolicyLoss) {
							totalPolicyLoss = ppoLoss;
							hasAnyPolicyLoss = true;
						} else {
							totalPolicyLoss += ppoLoss;
						}

						// Compute KL divergence & clip fraction
						{
							RG_NOGRAD;
							auto logRatio = logProbs - oldProbs;
							auto klTensor = (exp(logRatio) - 1) - logRatio;
							float kl = klTensor.mean().detach().cpu().item<float>();
							meanDivergence += kl;

							float clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat)).cpu().item<float>();
							clipFractions.push_back(clipFraction);
						}

						meanEntropy += entropy.cpu().detach().item<float>();
					}

					torch::Tensor valueLoss;
					if (trainCritic) {
						// Compute value loss
						vals = vals.view_as(targetValues);
						valueLoss = valueLossFn(vals, targetValues) * batchSizeRatio;
						
						// Accumulate value loss
						if (!hasAnyValueLoss) {
							totalValueLoss = valueLoss;
							hasAnyValueLoss = true;
						} else {
							totalValueLoss += valueLoss;
						}

						meanValLoss += valueLoss.cpu().detach().item<float>();
					}

					report.Accum("PPO Gradient Time", timer.Elapsed());
					numMinibatchIterations += 1;
				}
			}

			// Now perform backward pass once per batch with accumulated losses
			Timer gradTimer = {};
			if (hasAnyPolicyLoss)
				totalPolicyLoss.backward();
			if (hasAnyValueLoss)
				totalValueLoss.backward();
			report.Accum("PPO Total Backward Time", gradTimer.Elapsed());

			// Gradient noise measurement
			if (config.measureGradientNoise) {
				if (trainPolicy)
					noiseTrackerPolicy->Update(policy->seq);
				if (trainCritic)
					noiseTrackerValueNet->Update(valueNet->seq);
			}

			// Gradient clipping
			if (trainPolicy)
				nn::utils::clip_grad_norm_(policy->parameters(), 0.5f);
			if (trainCritic)
				nn::utils::clip_grad_norm_(valueNet->parameters(), 0.5f);
			
			// Optimizer steps
			if (trainPolicy)
				policyOptimizer->step();
			if (trainCritic)
				valueOptimizer->step();

			// Update half precision models if they exist
			if (policyHalf)
				_CopyModelParamsHalf(policy, policyHalf);
			if (valueNetHalf)
				_CopyModelParamsHalf(valueNet, valueNetHalf);
				
			numIterations += 1;
		}
	}

	numIterations = RS_MAX(numIterations, 1);
	numMinibatchIterations = RS_MAX(numMinibatchIterations, 1);

	// Compute averages for the metrics that will be reported
	meanEntropy /= numMinibatchIterations;
	meanDivergence /= numMinibatchIterations;
	meanValLoss /= numMinibatchIterations;
	meanRatio /= numMinibatchIterations;

	float meanClip = 0;
	if (!clipFractions.empty()) {
		for (float f : clipFractions)
			meanClip += f;
		meanClip /= clipFractions.size();
	}

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = _CopyParams(policy);
	auto criticAfter = _CopyParams(valueNet);

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	float totalTime = totalTimer.Elapsed();

	// Assemble and return report
	cumulativeModelUpdates += numIterations;
	report["PPO Batch Consumption Time"] = totalTime / numIterations;
	report["Cumulative Model Updates"] = cumulativeModelUpdates;
	report["Policy Entropy"] = meanEntropy;
	report["Mean KL Divergence"] = meanDivergence;
	report["Mean Ratio"] = meanRatio;
	report["Value Function Loss"] = meanValLoss;
	report["SB3 Clip Fraction"] = meanClip;
	report["Policy Update Magnitude"] = policyUpdateMagnitude;
	report["Value Function Update Magnitude"] = criticUpdateMagnitude;
	report["PPO Learn Time"] = totalTimer.Elapsed();

	if (config.measureGradientNoise) {
		if (noiseTrackerPolicy->lastNoiseScale != 0)
			report["Grad Noise Policy"] = noiseTrackerPolicy->lastNoiseScale;
		if (noiseTrackerValueNet->lastNoiseScale != 0)
			report["Grad Noise Value Net"] = noiseTrackerValueNet->lastNoiseScale;
	}

	policyOptimizer->zero_grad();
	valueOptimizer->zero_grad();
}

// Get sizes of all parameters in a sequence
std::vector<uint64_t> GetSeqSizes(torch::nn::Sequential& seq) {
	std::vector<uint64_t> result = {};

	for (int i = 0; i < seq->size(); i++)
		for (auto param : seq[i]->parameters())
			result.push_back(param.numel());

	return result;
}

constexpr const char* MODEL_FILE_NAMES[] = {
		"PPO_POLICY.lt",
		"PPO_CRITIC.lt",
};

constexpr const char* OPTIM_FILE_NAMES[] = {
	"PPO_POLICY_OPTIM.lt",
	"PPO_CRITIC_OPTIM.lt",
};

void TorchLoadSaveSeq(torch::nn::Sequential seq, std::filesystem::path path, c10::Device device, bool load) {
	if (load) {
		auto streamIn = std::ifstream(path, std::ios::binary);
		streamIn >> std::noskipws;

		if (!streamIn.good())
			RG_ERR_CLOSE("Failed to load from " << path << ", file does not exist or can't be accessed");

		auto sizesBefore = GetSeqSizes(seq);

		try {
			torch::load(seq, streamIn, device);
		} catch (std::exception& e) {
			RG_ERR_CLOSE(
				"Failed to load model, checkpoint may be corrupt or of different model arch.\n" <<
				"Exception: " << e.what()
			);
		}

		// Torch will happily load in a model of a totally different size, then we will crash when we try to use it
		// So we need to manually check if it is the same size
		auto sizesAfter = GetSeqSizes(seq);
		if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(), sizesAfter.end())) {
			std::stringstream stream;
			stream << "Saved model has different size than current model, cannot load model from " << path << ":\n";
			
			for (int i = 0; i < 2; i++) {
				stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
				for (uint64_t size : (i ? sizesAfter : sizesBefore))
					stream << size << ' ';

				stream << " ]";
				if (i == 0)
					stream << ",\n";
			}

			RG_ERR_CLOSE(stream.str());
		}

	} else {
		auto streamOut = std::ofstream(path, std::ios::binary);
		torch::save(seq, streamOut);
	}
}

void TorchLoadSaveAll(RLGPC::PPOLearner* learner, std::filesystem::path folderPath, bool load) {

	if (load) {
		if (!std::filesystem::exists(folderPath / MODEL_FILE_NAMES[0]))
			RG_ERR_CLOSE("PPOLearner: Failed to find file \"" << MODEL_FILE_NAMES[0] << "\" in " << folderPath << ".")
	}

	TorchLoadSaveSeq(learner->policy->seq, folderPath / MODEL_FILE_NAMES[0], learner->device, load);

	if (!load || std::filesystem::exists(folderPath / MODEL_FILE_NAMES[1]))
		TorchLoadSaveSeq(learner->valueNet->seq, folderPath / MODEL_FILE_NAMES[1], learner->device, load);

	if (load) {
		if (learner->policyHalf)
			_CopyModelParamsHalf(learner->policy, learner->policyHalf);
		if (learner->valueNetHalf)
			_CopyModelParamsHalf(learner->valueNet, learner->valueNetHalf);
	}

	// Load or save optimizers
	if (load) {
		try {
			for (int i = 0; i < 2; i++) {
				auto path = folderPath / OPTIM_FILE_NAMES[i];

				if (!std::filesystem::exists(path)) {
					RG_LOG("WARNING: No optimizer found at " << path << ", optimizer will be reset");
					continue;
				}

				{ // Check if empty
					std::ifstream testStream = std::ifstream(path, std::istream::ate | std::ios::binary);
					if (testStream.tellg() == 0) {
						RG_LOG("WARNING: Saved optimizer is empty, optimizer will be reset");
						continue;
					}
				}

				DataStreamIn in = DataStreamIn(path, false);

				auto& optim = i ? learner->valueOptimizer : learner->policyOptimizer;

				torch::serialize::InputArchive policyOptArchive;
				policyOptArchive.load_from(path.string(), learner->device);
				(i ? learner->valueOptimizer : learner->policyOptimizer)->load(policyOptArchive);
			}

		} catch (std::exception& e) {
			RG_ERR_CLOSE(
				"Failed to load optimizers, exception: " << e.what() << "\n" <<
				"Checkpoint may be corrupt."
			);
		}
	} else {
		for (int i = 0; i < 2; i++) {
			torch::serialize::OutputArchive policyOptArchive;
			(i ? learner->valueOptimizer : learner->policyOptimizer)->save(policyOptArchive);
			policyOptArchive.save_to((folderPath / OPTIM_FILE_NAMES[i]).string());
		}
	}
}

void RLGPC::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	RG_LOG("PPOLearner(): Saving models to: " << folderPath);
	TorchLoadSaveAll(this, folderPath, false);
}

RLGPC::DiscretePolicy* RLGPC::PPOLearner::LoadAdditionalPolicy(std::filesystem::path folderPath) {
	std::filesystem::path policyPath = folderPath / MODEL_FILE_NAMES[0];
	if (!std::filesystem::exists(policyPath))
		return NULL;

	RLGPC::DiscretePolicy* newPolicy = new RLGPC::DiscretePolicy(policy->inputAmount, policy->actionAmount, policy->layerSizes, policy->device);
	TorchLoadSaveSeq(newPolicy->seq, policyPath, newPolicy->device, true);
	return newPolicy;
}

void RLGPC::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	RG_LOG("PPOLearner(): Loading models from: " << folderPath);
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	TorchLoadSaveAll(this, folderPath, true);

	UpdateLearningRates(config.policyLR, config.criticLR);
}

void RLGPC::PPOLearner::UpdateLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	for (auto& g : policyOptimizer->param_groups())
		static_cast<torch::optim::AdamOptions&>(g.options()).lr(policyLR);

	for (auto& g : valueOptimizer->param_groups())
		static_cast<torch::optim::AdamOptions&>(g.options()).lr(criticLR);

	std::stringstream updatedMsg;
	updatedMsg << std::scientific << "Updated learning rate to [" << policyLR << ", " << criticLR << "]";
	RG_LOG("PPOLearner: " << updatedMsg.str());
}