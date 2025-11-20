# Enter your code here. Read input from STDIN. Print output to STDOUT

# Given data
l_1 = [15, 12, 8, 8, 7, 7, 7, 6, 5, 3]
l_2 = [10, 25, 17, 11, 13, 17, 20, 13, 9, 15]

# Calculate means
mean_1 = sum(l_1) / len(l_1)
mean_2 = sum(l_2) / len(l_2)

# Calculate numerator and denominator for Pearson's correlation
numerator = sum((l_1[i] - mean_1) * (l_2[i] - mean_2) for i in range(len(l_1)))
denominator = (sum((l_1[i] - mean_1) ** 2 for i in range(len(l_1))) * sum((l_2[i] - mean_2) ** 2 for i in range(len(l_2)))) ** 0.5

r = numerator / denominator

# Print result rounded to 3 decimal places
print(f"{r:.3f}")
