from src.Model_oo import Model

model = Model([5, 2])
print(model.params)
model.save_params('oo.pickle')

model_2 = Model([1, 5])

# print(model_2.params)
model_2.load_params('o9o.pickle')
print(model_2.params)
