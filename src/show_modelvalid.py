import pickle


with open("./model_valid.bin", "rb") as f:
    results2 = pickle.load(f)
    print(results2)
for tuple in results2:
    print(tuple)