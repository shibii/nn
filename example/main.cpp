#include <dtnn.hpp>
#include <ctime>

int main(unsigned int argc, const char* argv[]) {
  using namespace dtnn;

  af::setSeed(time(NULL));

  af::array inputs = af::randu(af::dim4(1, 1, 1, 10000)) * af::Pi;
  af::array targets = af::sin(inputs);


  auto optimizer = std::make_shared<SGD>(1e-3f);
  auto provider = TrainingBlob(inputs.host<float>(), targets.host<float>(),
                               inputs.dims(), targets.dims());

  Network nn(provider, optimizer);

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


  while (true) {
    for (int i = 0; i < 10000; i++) {
      nn.train(provider, 16);
    }
    nn.test(provider, 10000);
    auto realtargets = provider.current_batch_.targets;
    auto predtargets = loss->output_;
    std::cout << af::sum<float>(realtargets - predtargets) / 10000.f << std::endl;
  }

  return 0;
}