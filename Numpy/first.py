# import numpy as np

# arr = np.array([1, 2, 3, 4, 5])
# print(arr[len(arr) - 1])

# arr_2D = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# print(arr_2D)
# print(arr_2D.shape)
# print(arr_2D.size)
# print(arr_2D[0][len(arr_2D[0]) - 1])

# zeroes = np.zeros(5)
# print(zeroes)

# ones = np.ones(5)
# print(ones)

# zeroes = np.ones((3, 4))
# print(zeroes)

# full = np.full((3, 3), 107)
# print(full)

# range = np.arange(1, 10, 2)
# print(range)

# linear_arr = np.linspace(0, 1, 10)
# print(linear_arr)

# random_arr = np.random.rand(5, 5)
# print(random_arr)

# reshape_arr = np.arange(0, 10, 1).reshape((5, 2))
# print(reshape_arr)

# flatten_arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).flatten()
# print(flatten_arr)

# x = np.array([1, 2, 3, 4])
# y = np.array([5, 6, 7, 8])
# concate = np.concatenate((x, y))
# print(concate)

# sum_arr = np.arange(1, 101, 1).sum()
# print(sum_arr)

# mean_arr = np.arange(1, 101, 1).mean()
# print("Mean", mean_arr)

# random_arr = np.random.rand(3, 5).max()
# print(random_arr)

# random_arr = np.random.rand(3, 5).min()
# print(random_arr)

# range_arr = np.arange(1, 10, 2)
# sqrt_arr = np.sqrt(range_arr)
# print(sqrt_arr)

# slicing

# random_arr = np.arange(0, 1000, 10)
# print(random_arr[10:])

# Sorting

# arr = np.random.rand(4)
# sorted_arr = np.sort(arr)
# print(sorted_arr)

# arr = np.arange(11, 101, 1)
# where_method = np.where(arr > 95)
# print(where_method)

# arr = np.arange(1, 11, 1)
# square = np.square(arr).sum()
# print(square)

# arr = np.full(5, 10000000000)
# log_arr = np.log10(arr)
# print(log_arr)

# arr = np.arange(1, 10, 1)
# appended_arr = np.append(arr, 10)
# print(appended_arr)

# identity_amtrix = np.eye(3)
# print(identity_amtrix)

# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# arr1 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
# dot_product = np.dot(arr, arr1)
# print(dot_product)

# arr = np.array([[1, 2.34, 3], [4, 5, 6], [7, 8, 9]])
# print(arr.ndim)
# print(arr.dtype)
# print(arr)

# arr = np.array([1.23, 4.56, 7.89]) 
# int_arr = arr.astype(str)
# print(int_arr)

# arr = np.array([1, 2, 3, 4, 5])
# print(arr / 10)

# Aggreate functions
# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr.transpose())

# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr.T)
# print(np.var(arr))

# Array Properties & operations
# arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# print(arr.reshape(9, 1))

# neg_arr = np.array([-1, -2, -3, -4, -5])
# abs_arr = np.abs(neg_arr)
# print(abs_arr)

# sqrt_arr = np.array([1, 4, 9, 16, 25])
# sqrt_result = np.sqrt(sqrt_arr) 
# print(sqrt_result)

# filtering

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# filtered_arr = arr[arr > 5]
# print(filtered_arr)

# arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
# new_arr = np.insert(arr, 5, 99)
# print(new_arr)

# arr = np.array([1, 2, 3, 4, 5])
# new_arr = np.append(arr, 100) 
# print(new_arr)

# arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# new_arr = np.insert(arr_2d, 0, [100, 200, 300], axis=1)
# print(new_arr)

# arr = np.array([12, 34, 56, 78, 90, 123, 456, 789, 1000])
# new_arr = np.delete(arr, len(arr) - 1)
# print(new_arr)

# arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# new_arr = np.delete(arr_2d, 0, axis=1)
# print(new_arr)

# Broadcasting

# arr = np.array([100, 200, 300, 400, 500])
# print(arr - (arr / 10))

# arr1 = np.array([1, 2, 3, 4 ,5 ])
# arr2 = np.array([1, 2, 3, 4 ,5 ])
# print(arr1 * arr2)

# Handling missing data

# arr = np.array([1, 2, np.nan, 4, 5])
# arr = np.nan_to_num(arr, nan=100)
# print(arr)

# arr = np.array([1, 2, 3, np.inf, 5, -np.inf, 6, 7])
# print(np.isinf(arr))
# arr = np.nan_to_num(arr, posinf=1000, neginf=-1000)
# print(arr)