#include <dtnn.hpp>
#include <ctime>

int main(unsigned int argc, const char* argv[]) {
  using namespace dtnn;

  af::setSeed(time(NULL));

  SGD optimizer(1e-4f);
  Feed sample;
  sample.signal = af::randu(af::dim4(1)) * af::Pi;
  int units1 = 100;
  FullyConnected fc1(units1);
  auto w1 = fc1.init(sample);
  optimizer.attach(w1);
  fc1.forward(sample);
  Logistic af1;
  int units2 = 1;
  FullyConnected fc2(units2);
  auto w2 = fc2.init(sample);
  optimizer.attach(w2);
  Logistic af2;
  SquaredLoss loss;

  while (true) {
    float totalloss = 0;
    for (int i = 0; i < 1000; i++) {
      Feed f;
      af::array input = af::randu(af::dim4(1, 1, 1, 100)) * af::Pi;
      af::array target = af::sin(input);
      f.signal = input;

      fc1.forward(f);
      af1.forward(f);
      fc2.forward(f);
      af2.forward(f);
      loss.error(f, target);

      af2.backward(f);
      fc2.backward(f);
      af1.backward(f);
      fc1.backward(f);

      optimizer.optimize();
    }

    Feed f;
    af::array input = af::randu(af::dim4(1, 1, 1, 1000)) * af::Pi;
    af::array target = af::sin(input);
    f.signal = input;

    fc1.forward(f);
    af1.forward(f);
    fc2.forward(f);
    af2.forward(f);
    loss.error(f, target);
    std::cout << af::sum<float>(loss.output_ - target) / 1000.f << std::endl;
  }

  return 0;
}