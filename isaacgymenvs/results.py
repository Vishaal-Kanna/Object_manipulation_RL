from numpy import load

data = load('/home/vishaal/Downloads/evaluations.npz')
lst = data.files
for i in range(0,data['results'].shape[0]):
    print(data['results'][i][0])
