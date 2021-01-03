import numpy as np
def sigmoid(x):
	return 1/(1+np.exp(-x))
class Neuron:
	def __init__(self,weights): # Инициализация нейрона
		self.weights = weights
		self.output = 0 
	def activateFF(self,inp): # Активация каждого нейрона
		self.output = sigmoid(np.inner(inp,self.weights))
		return self.output
	def backTeach(self,error,inputs,learning_rate): # Обучение
		w_d = error*(self.output*(1-self.output)) 
		errors = []
		for a in range(len(self.weights)):
			self.weights[a] = self.weights[a]-inputs[a]*w_d*learning_rate # Learning rate
			errors.append(self.weights[a]*w_d)
		return errors
		
class Layer():
	def __init__(self,neuroWeights): # Создание слоя
		self.neurons = []
		self.out = []
		self.size = 0
		self.last_size_width = len(neuroWeights[0])
		for a in range(len(neuroWeights)):
			self.neurons.append(Neuron(neuroWeights[a]))
		self.size = len(self.neurons)
	def input(self,inp): # Активация всех нейронов
		self.out.clear()
		for a in range(self.size):
			self.out.append(self.neurons[a].activateFF(inp))
	def output(self): #Возвращаем результат активации нейронов
		return self.out
	def teachLayer(self,errors,inputs_l,learning_rate):# Для внутренних слоёв и выходных
		errors_b = np.array([0]*(self.last_size_width))
		for a in range(self.size):
			erN = self.neurons[a].backTeach(errors[a],inputs_l,learning_rate)
			errors_b = np.array(erN)+errors_b
		return errors_b
	def teachInputLayer(self,errors,inputs_l,learning_rate): # Для входного слоя
		errors_b = np.array([0]*(self.last_size_width))
		for a in range(self.size):
			erN = self.neurons[a].backTeach(self.out[a]-errors[a],inputs_l,learning_rate)
			errors_b = np.array(erN)+errors_b # Суммирование ошибок с разных нейронов
			
		return errors_b
class NeuronNetworkEE():
	def __init__(self,layers):
		self.layers = []
		self.count_l = len(layers)
		for a in range(self.count_l):
			self.layers.append(Layer(layers[a]))
	def FeedForward(self,inpt): # Функция прямого распространения
		self.layers[0].input(inpt)
		for a in range(1,self.count_l):
			self.layers[a].input(self.layers[a-1].output())
		return self.layers[self.count_l-1].output()
	def teachOne(self,inpt,target,learning_rate): # Фукнция одного прохода обучения
		self.FeedForward(inpt)
		er = self.layers[self.count_l-1].teachInputLayer(target,self.layers[self.count_l-2].output(),learning_rate)
		for a in range(2,self.count_l):
			er = self.layers[self.count_l-a].teachLayer(er,self.layers[self.count_l-a-1].output(),learning_rate)
		self.layers[0].teachLayer(er,inpt,learning_rate)
	def teach(self,iterations,inputs,targets,learning_rate):
		for a in range(iterations):
			for b in range(len(inputs)):
				self.teachOne(inputs[b],targets[b],learning_rate)
