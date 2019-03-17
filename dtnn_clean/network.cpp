#include "network.hpp"

namespace dtnn {
  Network::Network(std::shared_ptr<SampleProvider> provider) {
    provider_ = provider;
    currentbatch_ = provider_->batch(1);
  }
  void Network::add(std::shared_ptr<WeightlessStage> stage) {
    stages_.push_back(stage);
  }
  void Network::add(std::shared_ptr<WeightedStage> stage) {
    Feed feed;
    feed.signal = currentbatch_.inputs;
    for (auto stage : stages_) {
      stage->forward(feed);
    }
    optimizer_->attach(stage->init(feed));
    stages_.push_back(stage);
  }
  void Network::add(std::shared_ptr<LossFunction> loss) {
    loss_ = loss;
  }
  void Network::add(std::shared_ptr<Optimizer> optimizer) {
    optimizer_ = optimizer;
  }
  void Network::train(dim_t batchsize) {
    currentbatch_ = provider_->batch(batchsize);
    Feed feed;
    feed.signal = currentbatch_.inputs;
    for (auto stage : stages_) {
      stage->forward(feed);
    }
    loss_->error(feed, currentbatch_.targets);
    for (auto stage = stages_.rbegin(); stage != stages_.rend(); stage++) {
      (*stage)->backward(feed);
    }
    optimizer_->optimize();
  }
  void Network::predict(dim_t batchsize) {
    currentbatch_ = provider_->batch(batchsize);
    Feed feed;
    feed.signal = currentbatch_.inputs;
    for (auto stage : stages_) {
      stage->forward(feed);
    }
    loss_->output(feed);
  }
}