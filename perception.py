# Inputs: [has ID, on the list]
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
# Correct answers for AND: only both = 1
answers = [0, 1, 1, 0]

weights = [0.0, 0.0]
bias = 0.0
learning_rate = 0.1

def predict(guest):
    total = guest[0] * weights[0] + guest[1] * weights[1] + bias
    if total > 0:
        return 1
    else:
        return 0

for round in range(10):
    for guest, answer in zip(inputs, answers):
        guess = predict(guest)
        error = answer - guess
        weights[0] += learning_rate * error * guest[0]
        weights[1] += learning_rate * error * guest[1]
        bias += learning_rate * error
    print("Round", round, "weights:", weights, "bias:", bias)

print("Final test:")
for guest in inputs:
    print(guest, "->", predict(guest))
