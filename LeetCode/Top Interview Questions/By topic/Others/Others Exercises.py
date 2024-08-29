'''
CHALLENGES INDEX

1. Two Sum (Arrays) (Hash Tables)
4. Median of Two Sorted Arrays (Arrays) 


(xx)

'''




'1. Two Sum'

# # 1st approach
# nums = [3,2,4]
# target = 6
# result = []

# for i in range(len(nums)):

#     for j in range(i+1, len(nums)):

#         if nums[i] + nums[j] == target:
#             result = [i, j]
#             break
        
#     if result:
#         break
     
# print(result)


# # 2nd approach
# nums = [3,2,4]
# target = 6
# hashmap = {v:k for k,v in enumerate(nums)}
# result = []

# for i in range(len(nums)):

#     comp = target - nums[i]

#     if comp in hashmap and i != hashmap[comp]:
#         result = [i, hashmap[comp]]
#         break
   
 
# print(result)


# # 3rd approach
# nums = [3,2,4]
# target = 6
# hashmap = {}
# result = []

# for i in range(len(nums)):

#     comp = target - nums[i]

#     if comp in hashmap:
#         result = [i, hashmap[comp]]
#         break

#     hashmap[nums[i]] = i
    
# print(result)




'2. Add Two Numbers'