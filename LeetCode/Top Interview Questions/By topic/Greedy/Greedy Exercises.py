'''
CHALLENGES INDEX

11. Container With Most Water (Array) (TP) (GRE)
55. Jump Game (Array) (DP) (GRE)


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


(XX)

'''


'11. Container With Most Water'
# def x():

#     # Input
#     heights = [1,8,6,2,5,4,8,3,7]


#     # My Approach
#     max_area = 0

#     for i in range(len(heights)):

#         for j in range(i+1, len(heights)):

#             height = min(heights[i], heights[j])
#             width = j-i
#             area = height * width

#             max_area = max(max_area, area)

#     print(max_area)


#     '''
#     Note:
#         While this approach works, its complexity goes up to O(n), and is required to be more efficient
#     '''


#     # Two-pointer solution

#     left = 0
#     right = len(heights)-1
#     max_area = 0

#     while left < right:

#         h = min(heights[left], heights[right])
#         width = right - left
#         area = h * width

#         max_area = max(max_area, area)

#         if heights[left] <= heights [right]:
#             left += 1
        
#         else:
#             right -= 1


#     print(max_area)


'55. Jump Game'
# def x():

#     # Input
#     # Case 1
#     nums = [2,3,1,1,4]
#     # Output: True

#     # Case 2
#     nums = [3,2,1,0,4]
#     # Output: False

#     # Custom Case
#     nums = [2,0,0]
#     # Output: 

#     '''
#     My Approach

#         Intuition - Brute Force:

#             - I will check item by item to determine if the end of the list is reachable

#     '''
#     def canJump(nums:list[int]) -> bool:

#         # Corner case: nums.lenght = 1 / nums[0] = 0
#         if len(nums) == 1 and nums[0] == 0:
#             return True
        
#         idx = 0

#         while True:

#             idx += nums[idx]

#             if idx >= len(nums)-1:
#                 return True
            
#             if nums[idx] == 0 and idx < len(nums)-1:
#                 return False
            
#     # Testing
#     print(canJump(nums))

#     'Notes: This solution suffice 91,2% of the case'


#     'Backtrack Approach'
#     def canJump(nums: list[int]) -> bool:

#         if len(nums)==1:
#             return True  

#         #Start at num[-2] since nums[-1] is given
#         backtrack_index = len(nums)-2 
#         #At nums[-2] we only need to jump 1 to get to nums[-1]
#         jump =1  

#         while backtrack_index>0:
#             #We can get to the nearest lily pad
#             if nums[backtrack_index]>=jump: 
#                 #now we have a new nearest lily pad
#                 jump=1 
#             else:
#                 #Else the jump is one bigger than before
#                 jump+=1 
#             backtrack_index-=1
        
#         #Now that we know the nearest jump to nums[0], we can finish
#         if jump <=nums[0]: 
#             return True
#         else:
#             return False 

#     'Notes: Right now I am not that interested in learning bactktracking, that will be for later'













