'''
CHALLENGES INDEX

238. Product of Array Except Self (PS)


*LL: Linked-Lists
*BS: Binary Search
*DP: Dynamic Programming
*RC: Recursion
*TP: Two-pointers
*FCD: Floyd's cycle detection (Hare & Tortoise approach)
*PS: Preffix-sum
*SW: Sliding-Window
*MEM: Memoization
*GRE: Greedy
*DQ: Divide and Conquer
*BT: Backtracking
*BFS & DFS: Breadth-First Search & Depth-First Search
*Arrays, Hash Tables & Matrices
*Sorting
*Heaps, Stacks & Queues
*Graphs & Trees
*Others


(1)
'''


'''238. Product of Array Except Self'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,3,4]
#     # Output: [24,12,8,6]

#     # Case 2
#     nums = [-1,1,0,-3,3]
#     # Output: [0,0,9,0,0]


#     'My Approach / Brute forcing'
#     from math import prod

#     def product_except_self(nums:list[int]) -> list[int]:

#         res = []

#         for i in range(len(nums)):
#             res.append(prod(nums[:i]+nums[i+1:]))

#         return res

#     # Testing
#     print(product_except_self(nums=nums))

#     'Note: This solution suffices 75% of test cases, but resulted inefficient with large inputs'


#     'My Approach v2 - (Trying to carry the result - pseudo-prefix sum)'
#     from math import prod

#     def product_except_self(nums:list[int]) -> list[int]:

#         res = []
#         nums_prod = prod(nums)

#         for i in range(len(nums)):

#             elem = nums_prod//nums[i] if nums[i] != 0 else prod(nums[:i]+nums[i+1:])
#             res.append(elem)


#         return res

#     # Testing
#     print(product_except_self(nums=nums))

#     'Note: It worked and beated 85% in time compl. and 74% in memory compl.'


#     '''
#     Preffix-Suffix product Approach - Customary

#     Intuition:
#         - The core idea of this approach is to build an array of the carrying product of all elements from left to right (Preffix)
#             and build another more array with the sabe but from right to left (suffix).

#         - After having that by combining those products but element by element, the preffix from 0 to n-1 indexed and the suffix from n-1 to 0 indexed
#             and EXCLUDING the current index (i) in the final traversal, the 'self' element is explicitly excluded from the product.
#     '''

#     from itertools import accumulate
#     import operator

#     def product_except_self(nums:list[int]) -> list[int]:

#         res = []

#         # Populate both preffix and suffix
#         preffix = list(accumulate(nums, operator.mul))
#         suffix = list(accumulate(reversed(nums), operator.mul))[::-1]


#         # Combine the results
#         for i in range(len(preffix)):

#             if 0 < i < len(preffix)-1:
#                 res.append(preffix[i-1]*suffix[i+1])
            
#             elif i == 0:
#                 res.append(suffix[i+1])

#             else:
#                 res.append(preffix[i-1])
        
#         return res

#     # Testing
#     print(product_except_self(nums=nums))

#     'Done'
































