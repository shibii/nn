#include "network.hpp"

namespace dtnn {
  Network::Network(af::dim4 input_dimensions, std::shared_ptr<Optimizer> optimizer)
    : optimizer_(optimizer), inputdim_(input_dimensions)
  {
  }
  void Network::add(std::shared_ptr<WeightlessStage> stage) {
    stages_.push_back(stage);
  }
  void Network::add(std::shared_ptr<WeightedStage> stage) {
    Feed feed;
    feed.signal = af::constant(0.f, inputdim_);
    for (auto &stage : stages_) {
      stage->forward(feed);
    }
    optimizer_->attach(stage->init(feed));
    stages_.push_back(stage);
  }
  void Network::add(std::shared_ptr<LossFunction> loss) {
    loss_ = loss;
  }
  void Network::train(TrainingBatch &batch) {
    Feed feed;
    feed.signal = batch.inputs;
    for (auto &stage : stages_) {
      stage->forward(feed);
    }
    feed.signal = loss_->error(feed, batch.targets);
    for (auto stage = stages_.rbegin(); stage != stages_.rend(); stage++) {
      (*stage)->backward(feed);
    }
    optimizer_->optimize();
  }
  TestingResult Network::test(TrainingBatch &batch) {
    Feed feed;
    feed.signal = batch.inputs;
    for (auto &stage : stages_) {
      stage->forward(feed);
    }
    af::array loss = loss_->loss(feed, batch.targets);
    af::array output = loss_->output(feed);
    return TestingResult(output, batch.targets, loss);
  }
  af::array Network::predict(PredictionBatch &batch) {
    Feed feed;
    feed.signal = batch.inputs;
    for (auto &stage : stages_) {
      stage->forward(feed);
    }
    return loss_->output(feed);
  }
}