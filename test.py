from collections import Counter
def mode2(sample):
	c = Counter(sample)
	value = []
	amount = []
	for entry in c.most_common(1):
		value.append(entry[0])
		amount.append(entry[1]/len(sample))
	return value, amount



my_list = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 5, 6, 5, 6]
value, amount = mode2(my_list)
print(value)
print(amount)