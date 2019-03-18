#include "network.hpp"

namespace dtnn {
  Network::Network(SampleProvider &provider, std::shared_ptr<Optimizer> optimizer) {
    inputdim_ = provider.input_dimensions();
    optimizer_ = optimizer;
  }
  void Network::add(std::shared_ptr<WeightlessStage> stage) {
    stages_.push_back(stage);
  }
  void Network::add(std::shared_ptr<WeightedStage> stage) {
    Feed feed;
    feed.signal = af::constant(0.f, inputdim_);
    for (auto stage : stages_) {
      stage->forward(feed);
    }
    optimizer_->attach(stage->init(feed));
    stages_.push_back(stage);
  }
  void Network::add(std::shared_ptr<LossFunction> loss) {
    loss_ = loss;
  }
  void Network::train(TrainingProvider &provider, dim_t batchsize) {
    TrainingBatch batch = provider.batch(batchsize);
    Feed feed;
    feed.signal = batch.inputs;
    for (auto stage : stages_) {
      stage->forward(feed);
    }
    feed.signal = loss_->error(feed, batch.targets);
    for (auto stage = stages_.rbegin(); stage != stages_.rend(); stage++) {
      (*stage)->backward(feed);
    }
    optimizer_->optimize();
  }
  af::array Network::test(TrainingProvider &provider, dim_t batchsize) {
    TrainingBatch batch = provider.batch(batchsize);
    Feed feed;
    feed.signal = batch.inputs;
    for (auto stage : stages_) {
      stage->forward(feed);
    }
    return loss_->loss(feed, batch.targets);
  }
  af::array Network::predict(PredictionProvider &provider, dim_t batchsize) {
    PredictionBatch batch = provider.batch(batchsize);
    Feed feed;
    feed.signal = batch.inputs;
    for (auto stage : stages_) {
      stage->forward(feed);
    }
    return loss_->output(feed);
  }
}