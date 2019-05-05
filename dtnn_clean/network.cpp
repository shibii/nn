#include "network.hpp"

namespace dtnn {
  Network::Network(af::dim4 input_dimensions, std::shared_ptr<Optimizer> optimizer)
    : inputdim_(input_dimensions), optimizer_(optimizer)
  {
  }
  void Network::add(std::shared_ptr<WeightlessStage> stage) {
    stages_.push_back(stage);
  }
  void Network::add(std::shared_ptr<WeightedStage> stage) {
    Feed feed;
    feed.signal = af::constant(0.f, inputdim_);
    forward_stages(feed);
    optimizer_->attach(stage->init(feed.signal.dims()));
    stages_.push_back(stage);
  }
  void Network::add(std::shared_ptr<LossFunction> loss) {
    loss_ = loss;
  }
  void Network::generate_gradient(TrainingBatch &batch) {
    Feed feed;
    feed.signal = batch.inputs;
    forward_stages(feed);
    feed.signal = loss_->error(feed, batch.targets);
    backward_stages(feed);
  }
  void Network::update_weights() {
    optimizer_->optimize();
  }
  void Network::train(TrainingBatch &batch) {
    generate_gradient(batch);
    update_weights();
  }
  TestingResult Network::test(TrainingBatch &batch) {
    Feed feed;
    feed.signal = batch.inputs;
    forward_stages(feed);
    af::array loss = loss_->loss(feed, batch.targets);
    af::array output = loss_->output(feed);
    return TestingResult(output, batch.targets, loss);
  }
  PredictionResult Network::predict(PredictionBatch &batch) {
    Feed feed;
    feed.signal = batch.inputs;
    forward_stages(feed);
    return PredictionResult(loss_->output(feed));
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