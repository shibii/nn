#include <dtnn.hpp>
#include <ctime>

int main(unsigned int argc, const char* argv[]) {
  using namespace dtnn;

  af::setSeed(time(NULL));

  int units1 = 400;
  auto w1 = std::make_shared<wb>(wb(af::dim4(units1, 1), units1, 3.6 / sqrtf(units1)));
  auto g1 = std::make_shared<wb>(wb(af::dim4(units1, 1), units1, 0.f));
  FullyConnected fc1(w1, g1);

  Logistic af1;

  int units2 = 400;
  auto w2 = std::make_shared<wb>(wb(af::dim4(1, units2), 1, 3.6 / sqrtf(units2)));
  auto g2 = std::make_shared<wb>(wb(af::dim4(1, units2), 1, 0.f));
  FullyConnected fc2(w2, g2);

  Logistic af2;

  SquaredLoss loss;
  SGD optimizer(1e-1f);
  optimizer.attach(w1, g1);
  optimizer.attach(w2, g2);
  while (true) {
    float totalloss = 0;
    for (int i = 0; i < 1000; i++) {
      Feed f;
      af::array input = af::randu(af::dim4(1, 1, 1, 10)) * af::Pi;
      af::array target = af::sin(input);
      f.signal = input;

      fc1.forward(f);
      af1.forward(f);
      fc2.forward(f);
      af2.forward(f);
      loss.error(f, target);
      totalloss += loss.loss(target);

      af2.backward(f);
      fc2.backward(f);
      af1.backward(f);
      fc1.backward(f);

      optimizer.optimize();
      g1->zero();
      g2->zero();
    }
    std::cout << totalloss << std::endl;
  }

  return 0;
}