#include <dtnn.hpp>
#include <ctime>

int main(unsigned int argc, const char* argv[]) {
  using namespace dtnn;

  af::setSeed(time(NULL));

  af::array inputs = af::randu(af::dim4(1, 1, 1, 10000)) * af::Pi;
  af::array targets = af::sin(inputs);

  auto provider = std::make_shared<BlobProvider>(BlobProvider(inputs.host<float>(),
    targets.host<float>(), inputs.dims(), targets.dims()));

  auto optimizer = std::make_shared<SGD>(1e-4f);

  Network nn(provider);
  nn.add(optimizer);
  auto fc1 = std::make_shared<FullyConnected>(100);
  nn.add(fc1);
  nn.add(std::make_shared<Logistic>());
  auto fc2 = std::make_shared<FullyConnected>(1);
  nn.add(fc2);
  nn.add(std::make_shared<Logistic>());
  auto loss = std::make_shared<SquaredLoss>();
  nn.add(loss);
  while (true){
    for (int i = 0; i < 10000; i++) {
      nn.train(16);
    }
    nn.predict(10000);
    std::cout << af::sum<float>(loss->output_ - nn.currentbatch_.targets) / 10000.f << std::endl;
    std::cout << nn.currentbatch_.targets.dims(3) << std::endl;
  }

  return 0;
}