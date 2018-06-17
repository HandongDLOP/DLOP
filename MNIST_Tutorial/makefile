debug_main_cpu:
	g++ -g -o testing -std=c++98 -D__DEBUG__ main.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/LossFunction.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork.cpp -lcudart -lcudnn -lpthread

debug_main_gpu:
	g++ -g -o testing -std=c++98 -D__CUDNN__ -D__DEBUG__ main.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/LossFunction.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork.cpp -lcudart -lcudnn -lpthread

main_cpu:
	g++ -g -o testing -std=c++98 main.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/LossFunction.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork.cpp -lcudart -lcudnn -lpthread

main_gpu:
	g++ -g -o testing -std=c++98 -D__CUDNN__ main.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/LossFunction.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork.cpp -lcudart -lcudnn -lpthread

example:
	g++ -g -o testing -std=c++98 -D__CUDNN__ example.cpp ../Header/Shape.cpp ../Header/Data.cpp ../Header/Tensor.cpp ../Header/Operator.cpp ../Header/LossFunction.cpp ../Header/Optimizer.cpp ../Header/NeuralNetwork.cpp -lcudart -lcudnn -lpthread
