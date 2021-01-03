import NeuroLib as ne
import random
def test1(): # Первый тест на обучение распознование небольших картинок
	layers = [[[random.random() for _ in range(25)],[random.random() for _ in range(25)],[random.random() for _ in range(25)],[random.random() for _ in range(25)],[random.random() for _ in range(25)]],
	[[random.random() for _ in range(5)],[random.random() for _ in range(5)],[random.random() for _ in range(5)]],
	[[0.3,0.8,0.416]]]
	ns = ne.NeuronNetworkEE(layers)
	one = [0,0,0,0,1,
		0,0,0,1,1,
		0,0,1,0,1,
		0,0,0,0,1,
		0,0,0,0,1]
	one1 = [0,0,1,1,0,
		0,0,1,1,0,
		0,1,0,1,0,
		0,0,0,1,0,
		0,0,0,1,0]
	two = [1,1,1,1,1,
		1,0,0,1,0,
		0,0,1,0,0,
		0,1,0,0,0,
		1,1,1,1,1]
	three = [1,1,1,1,1,
		1,0,0,0,1,
		0,0,1,1,1,
		0,0,0,0,1,
		1,1,1,1,1]
	ns.teach(40000,[one,one1,two,three],[[0.1],[0.1],[0.8],[0.5]],0.5);
	print(ns.FeedForward(one))
	print(ns.FeedForward(two))
	print(ns.FeedForward(three))
	print(ns.FeedForward(one1))
test1()	
