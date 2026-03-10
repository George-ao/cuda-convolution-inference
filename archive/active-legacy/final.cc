#include "ece408net.h"
#include <cstdlib>
#include <string>

void inference_only(int batch_size, const std::string& dataset_path) {

  std::cout<<"Loading fashion-mnist data...";
  MNIST dataset(dataset_path);
  dataset.read_test_data(batch_size);
  std::cout<<"Done"<<std::endl;
  
  std::cout<<"Loading model...";
  Network dnn = createNetwork_GPU();
  std::cout<<"Done"<<std::endl;

  dnn.forward(dataset.test_data);
  float acc = compute_accuracy(dnn.output(), dataset.test_labels);
  std::cout<<std::endl;
  std::cout<<"Test Accuracy: "<<acc<< std::endl;
  std::cout<<std::endl;
}

int main(int argc, char* argv[]) {
  
  int batch_size = 10000;
  std::string dataset_path = "/projects/bche/project/data/fmnist-86/";

  if (const char* env_path = std::getenv("FMNIST_PATH")) {
    dataset_path = env_path;
  }

  if(argc >= 2){
    batch_size = atoi(argv[1]);
  }
  if(argc >= 3){
    dataset_path = argv[2];
  }

  std::cout<<"Test batch size: "<<batch_size<<std::endl;
  std::cout<<"Dataset path: "<<dataset_path<<std::endl;
  inference_only(batch_size, dataset_path);

  return 0;
}
