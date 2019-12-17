#include <iostream>
#include <ctime>
#include <chrono>
#include <nn.hpp>

#include "arrayfire_util.hpp"
#include "mnistSampler.hpp"

int main(unsigned int argc, const char* argv[]) {
  using namespace nn;

  nn::set_backend(nn::BACKEND::OPENCL);
  nn::seed(time(NULL));
  nn::info();

  af::array training_images, training_targets;
  bool parse_result = MNIST::parseMnist("../../mnist/train-images.idx3-ubyte",
                    "../../mnist/train-labels.idx1-ubyte", training_images,
                    training_targets);
  if (!parse_result) {
    std::cout << "failed to parse MNIST" << std::endl;
    return EXIT_FAILURE;
   }
  auto training_provider =
      TrainingBatchProvider(training_images, training_targets);
  std::cout << "training samples: " << training_provider.sample_count()
            << std::endl;

  af::array test_images, test_targets;
  parse_result = MNIST::parseMnist("../../mnist/t10k-images.idx3-ubyte",
                    "../../mnist/t10k-labels.idx1-ubyte", test_images, test_targets);
  auto test_provider = TrainingBatchProvider(test_images, test_targets);
  if (!parse_result) {
    std::cout << "failed to parse MNIST" << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << "test samples: " << test_provider.sample_count() << std::endl;

  auto optimizer = std::make_shared<Momentum>();
  Network nn(training_provider.sample_dimensions(), optimizer);

  auto conv1 = std::make_shared<Convolutional>(3, 3, 1, 1, 0, 0, 20);
  nn.add(conv1);
  nn.add(std::make_shared<MaxPool>(2, 2, 2, 2, 0, 0));
  nn.add(std::make_shared<LReL>());

  auto conv2 = std::make_shared<Convolutional>(3, 3, 1, 1, 0, 0, 40);
  nn.add(conv2);
  nn.add(std::make_shared<MaxPool>(2, 2, 2, 2, 0, 0));
  nn.add(std::make_shared<LReL>());

  auto conv3 = std::make_shared<Convolutional>(3, 3, 1, 1, 1, 1, 60);
  nn.add(conv3);
  nn.add(std::make_shared<MaxPool>(2, 2, 2, 2, 0, 0));
  nn.add(std::make_shared<LReL>());

  auto fc1 = std::make_shared<FullyConnected>(64);
  nn.add(fc1);
  nn.add(std::make_shared<LReL>());

  auto out = std::make_shared<FullyConnected>(10);
  nn.add(out);

  auto classifier = std::make_shared<SoftmaxCrossEntropy>();
  nn.add(classifier);

  dim_t batch_size = 32;

  std::vector<long long> indices(training_provider.sample_count());
  std::iota(indices.begin(), indices.end(), 0);

  while (true) {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << std::endl << "training";

    std::shuffle(indices.begin(), indices.end(),
                 std::mt19937{std::random_device{}()});

    for (int i = 0; i < training_provider.sample_count(); i += batch_size) {
      std::vector<long long> batch_indices(indices.begin() + i,
                                       indices.begin() + i + batch_size);
      auto batch = training_provider.batch(batch_indices);

      try {
        Hyperparameters hp;
        hp.learningrate = 1e-2f;
        hp.weight_decay = 1e-7f;
        nn.train(batch, hp);
      } catch (af::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        throw;
      }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << std::endl
              << "execution time: " << duration.count() << " seconds";

    float loss = 0.f;
    float accuracy = 0.f;

    start = std::chrono::high_resolution_clock::now();
    std::cout << std::endl << "testing";
    int batches = 0;
    for (int i = 0; i < test_provider.sample_count(); i += 100) {
      auto batch = test_provider.batch(i, 100);
      auto result = nn.test(batch);
      batches++;
      loss += result.loss();
      accuracy += result.accuracy();
    }

    std::cout << std::endl
              << "loss: " << loss / (float)batches
              << " accuracy: " << accuracy / (float)batches;

    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << std::endl
              << "execution time: " << duration.count() << " seconds";

    std::ofstream ostream("network.xml", std::ios::binary);
    Serializer::serializeXML(ostream, nn);
  }

  return EXIT_SUCCESS;
}