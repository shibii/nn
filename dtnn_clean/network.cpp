#include "network.hpp"

namespace nn {
  Network::Network(af::dim4 input_dimensions, std::shared_ptr<Optimizer> optimizer)
    : inputdim_(input_dimensions), optimizer_(optimizer), batch_samples_(0)
  {
  }
  void Network::add(std::shared_ptr<WeightlessStage> stage) {
    stages_.push_back(stage);
  }
  void Network::add(std::shared_ptr<WeightedStage> stage) {
    Feed feed;
    feed.signal = af::constant(0.f, inputdim_);
    forward_stages(feed);
    auto parameters = stage->init(feed.signal.dims());
    weights_.push_back(parameters);
    optimizer_->attach(parameters);
    stages_.push_back(stage);
  }
  void Network::add(std::shared_ptr<LossFunction> loss) {
    loss_ = loss;
  }
  void Network::generate_gradient(TrainingBatch &batch) {
    Feed feed;
    feed.signal = batch.samples_;
    forward_stages(feed);
    feed.signal = loss_->error(feed, batch.targets_);
    backward_stages(feed);
    batch_samples_ += (unsigned int)feed.signal.dims(3);
  }
  void Network::update_weights(Hyperparameters hyperparameters) {
    hyperparameters.batch_size = batch_samples_;
    optimizer_->optimize(hyperparameters);
    batch_samples_ = 0;
  }
  void Network::train(TrainingBatch &batch, Hyperparameters hyperparameters) {
    generate_gradient(batch);
    update_weights(hyperparameters);
  }
  TestingResult Network::test(TrainingBatch &batch) {
    Feed feed;
    feed.signal = batch.samples_;
    forward_stages(feed);
    af::array loss = loss_->loss(feed, batch.targets_);
    af::array output = loss_->output(feed);
    return TestingResult(output, batch.targets_, loss);
  }
  PredictionResult Network::predict(PredictionBatch &batch) {
    Feed feed;
    feed.signal = batch.samples_;
    forward_stages(feed);
    return PredictionResult(loss_->output(feed));
  }
  void Network::merge_weights(Network &from, float bias) {
    for (int layer = 0; layer < weights_.size(); layer++) {
      weights_[layer]->weights = bias * weights_[layer]->weights
        + (1.f - bias) * from.get_weights()[layer]->weights;
    }
  }
  std::vector<std::shared_ptr<OptimizableWeights>> Network::get_weights() {
    return weights_;
  }
  void Network::forward_stages(Feed &feed) {
    for (auto &stage : stages_) {
      stage->forward(feed);
    }
  }
  void Network::backward_stages(Feed &feed) {
    for (auto stage = stages_.rbegin(); stage != stages_.rend(); stage++) {
      (*stage)->backward(feed);
    }
  }
}