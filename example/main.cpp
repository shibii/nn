#include <ctime>
#include <dtnn.hpp>

int main(unsigned int argc, const char* argv[]) {
  using namespace dtnn;

  af::setSeed(time(NULL));

  af::array training_inputs = af::randu(af::dim4(1, 1, 1, 10000)) * af::Pi;
  af::array training_targets = af::sin(training_inputs);

  af::array test_inputs = af::randu(af::dim4(1, 1, 1, 10000)) * af::Pi;
  af::array test_targets = af::sin(test_inputs);

  auto training_provider = TrainingBlob(training_inputs.host<float>(), training_targets.host<float>(),
    training_inputs.dims(), training_targets.dims());

  auto test_provider = TrainingBlob(test_inputs.host<float>(), test_targets.host<float>(),
    test_inputs.dims(), test_targets.dims());

  auto optimizer = std::make_shared<SGD>(1e-3f);

  Network nn(training_provider.input_dimensions(), optimizer);

  auto fc1 = std::make_shared<FullyConnected>(20);
  nn.add(fc1);

  nn.add(std::make_shared<Logistic>());

  auto fc2 = std::make_shared<FullyConnected>(20);
  nn.add(fc2);

  nn.add(std::make_shared<Logistic>());

  auto fc3 = std::make_shared<FullyConnected>(1);
  nn.add(fc3);

  //nn.add(std::make_shared<Logistic>());

  auto loss = std::make_shared<SquaredLoss>();
  nn.add(loss);

  auto test_batch = test_provider.batch(10000);
  af::array validation_loss = nn.test(test_batch);
  std::cout << "loss: " << af::sum<float>(validation_loss) << std::endl;

  for (int epochs = 0; epochs < 10; epochs++) {
    for (int i = 0; i < 1000; i++) {
      auto training_batch = training_provider.batch(16);
      nn.train(training_batch);
    }
    test_batch = test_provider.batch(10000);
    validation_loss = nn.test(test_batch);
    std::cout << "loss: " << af::sum<float>(validation_loss) << std::endl;
  }

  return 0;
}