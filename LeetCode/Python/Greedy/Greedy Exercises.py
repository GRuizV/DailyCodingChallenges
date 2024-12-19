'''
CHALLENGES INDEX

11. Container With Most Water (Array) (TP) (GRE)
55. Jump Game (Array) (DP) (GRE)
122. Best Time to Buy and Sell Stock II (Array) (DP) (GRE)
134. Gas Station (Array) (GRE)
179. Largest Number (Array) (Sorting) (GRE)
215. Kth Largest Element in an Array (Array) (Heap) (DQ) (Sorting)
218. The Skyline Problem (Heaps) (DQ)
334. Increasing Triplet Subsequence (Array) (GRE)

45. Jump Game II (Array) (GRE) (DP)
12. Integer to Roman (Hash Table) (GRE)



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


(10)
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
    
#     from typing import Optional

#     # Input
#     # Case 1
#     nums = [2,3,1,1,4]
#     # Output: True

#     # Case 2
#     nums = [3,2,1,0,4]
#     # Output: False

#     '''
#     My Approach (DFS)

#         Intuition:

#             - Serialize 'nums' with enumerate in a 's_nums' holder.
#             - Initialize a list 'stack' initialized in the first element of 's_nums'         
#             - In a while loop ('stack exists'):
#                 + Initialize a position holder 'curr' with stack.pop(0).
#                 + if 'curr' holder first element (item position) added to its second element (item value) reaches the end of the list return True
#                     if ['curr'[0] + 'curr'[1]] >= len(nums):
#                         * Return True
#                 + Extend the stack to all reachable positions ( stack.extend([s_nums[curr[0]+x] for x in range(1, curr[1]+1)]) if curr[0]+x <= len(nums))

#             - If the code gets out of the loop, means the end of the list is out of reach, return False then.
#     '''

#     def canJump(nums: list[int]) -> bool:

#         # Serialize 'nums' in 's_nums'
#         s_nums = [(i,v) for i,v in enumerate(nums)]

#         # Initialize an stack list at the first s_num value
#         stack = [s_nums[0]]
    
#         while stack:
            
#             curr = stack.pop(0)

#             if curr[0]+curr[1] >= len(nums)-1:
#                 return True
           
#             stack.extend([s_nums[curr[0]+x] for x in range(1, curr[1]+1) if curr[0]+x <= len(nums) and s_nums[curr[0]+x] not in stack] )

#         # Return False if the code gets up to here
#         return False

#     # Testing
#     print(canJump(nums = nums))

#     '''Note: DFS only solved 45% of test cases and prooved to be inefficient O(2^n) exponentially complex'''




#     '''
#     Customary solution (Greedy)

#         Explanation:

#             - Keep track of the farthest index that can be reached.
#             - If at any point the current index exceeds this farthest index, it means the end of the array is not reachable.
#     '''

#     def canJump(nums: list[int]) -> bool:

#         # Initialize a int 'max_reachble' holder at 0
#         max_reachable = 0

#         # Iterate linearly the input
#         for i, num in enumerate(nums):

#             # If the current index 'i' exceeds the max_reachable, getting to the end is impossible
#             if i > max_reachable:
#                 return False
            
#             # Update the max_reachable with the current index max reachable
#             max_reachable = max(max_reachable, i+num)

#             # If the max reachable can at least the to the input's end, return True
#             if max_reachable >= len(nums)-1:
#                 return True

#         # Return False if the code gets up to here
#         return False

#     # Testing
#     print(canJump(nums = nums))

#     '''Note: This is the most elegant solution of all, solve this in O(n) time and O(1) space'''




#     '''
#     Customary solution (Greedy)

#         Explanation:

#             - Keep track of the farthest index that can be reached.
#             - If at any point the current index exceeds this farthest index, it means the end of the array is not reachable.
#     '''

#     def canJump(nums: list[int]) -> bool:

#         # Initialize a int 'max_reachble' holder at 0
#         max_reachable = 0

#         # Iterate linearly the input
#         for i, num in enumerate(nums):

#             # If the current index 'i' exceeds the max_reachable, getting to the end is impossible
#             if i > max_reachable:
#                 return False
            
#             # Update the max_reachable with the current index max reachable
#             max_reachable = max(max_reachable, i+num)

#             # If the max reachable can at least the to the input's end, return True
#             if max_reachable >= len(nums)-1:
#                 return True

#         # Return False if the code gets up to here
#         return False

#     # Testing
#     print(canJump(nums = nums))

#     '''Note: This is the most elegant solution of all, solve this in O(n) time and O(1) space'''




#     '''
#     A DP solution (Dynamic Programming)

#         Explanation:

#             1. Reachability State:
#                 - Use a boolean array dp where dp[i] is True if the index i is reachable from the starting index.
            
#             2. Base Case:
#                 - The first index dp[0] is always reachable because we start there.
            
#             3. Transition:
#                 - For each index i, if dp[i] is True, then all indices within the range [i+1, i+nums[i]] are also reachable. Update dp[j] to True for all such j.
            
#             4. Final Answer:
#                 - The value at dp[-1] (last index) will tell if the last index is reachable.
#     '''

#     def canJump(nums: list[int]) -> bool:

#         # Initialize boolean DP array
#         dp = [False]*len(nums)

#         # First item will be always reachable
#         dp[0] = True

#         # Process the input
#         for i in range(len(nums)):

#             if dp[i]:
                
#                 for j in range(1, nums[i]+1):
                    
#                     if i+j < len(nums):
#                         dp[i+j] = True 
        
#         # Return the last value of the DP table
#         return dp[-1]


#     # Testing
#     print(canJump(nums = nums))

#     '''Note: This is the most elegant solution of all, solve this in O(n^2) time and O(n) space'''

'''122. Best Time to Buy and Sell Stock II'''
# def x():

#     #Input
#     #Case 1
#     prices = [7,1,5,3,6,4]
#     #Output: 7

#     #Case 2
#     prices = [1,2,3,4,5]
#     #Output: 4

#     #Case 3
#     prices = [7,6,4,3,1]
#     #Output: 0

#     #Custom Case
#     prices = [3,3,5,0,0,3,1,4]
#     #Output: 0


#     'My approach'
#     def maxProfit(prices:list[int]) -> int:

#         if prices == sorted(prices, reverse=True):
#             return 0
        
#         buy = prices[0]
#         buy2 = None
#         profit1 = 0
#         profit2 = 0
#         total_profit = 0

#         for i in range(1, len(prices)):

#             if prices[i] < buy:
#                 buy = prices[i]
            
#             elif prices[i] - buy >= profit1:            
#                 profit1 = prices[i] - buy
#                 buy2 = prices[i] 

#                 for j in range(i+1, len(prices)):

#                     if prices[j] < buy2:
#                         buy2 = prices[j]

#                     elif prices[j] - buy2 >= profit2:
#                         profit2 = prices[j] - buy2
#                         total_profit = max(total_profit, profit1 + profit2)
            
#             total_profit = max(total_profit, profit1)

#         return total_profit

#     # Testing
#     print(maxProfit(prices=prices))

#     'This solution went up to solve 83% of the cases, the gap was due to my lack of understanding of the problem'


#     '''Same Kadane's but modified'''
#     def maxProfit(prices:list[int]) -> int:

#         max = 0 
#         start = prices[0]
#         len1 = len(prices)

#         for i in range(0 , len1):

#             if start < prices[i]: 
#                 max += prices[i] - start

#             start = prices[i]

#         return max

#     # Testing
#     print(maxProfit(prices=prices))

#     'My mistake was to assume it can only be 2 purchases in the term, when it could be as many as it made sense'

'''134. Gas Station'''
# def x():

#     # Input
#     #Case 1
#     gas, cost = [1,2,3,4,5], [3,4,5,1,2]
#     #Output = 3

#     #Case 2
#     gas, cost = [2,3,4], [3,4,3]
#     #Output = -1

#     # #Custom Case 
#     gas, cost = [3,1,1], [1,2,2]
#     #Output = 0


#     '''
#     My Approach

#         Intuition:
#             - Handle the corner case where sum(gas) < sum(cos) / return -1
#             - Collect the possible starting point (Points where gas[i] >= cost[i])
#             - Iterate to each starting point (holding it in a placeholder) to check 
#                 if a route starting on that point completes the lap:
                
#                 - if it does: return that starting point
#                 - if it doesn't: jump to the next starting point

#             - If no lap is completed after the loop, return -1.
#     '''

#     def canCompleteCircuit(gas:list[int], cost:list[int]) -> int:
        
#         # Handle the corner case
#         if sum(gas) < sum(cost):
#             return -1
        
#         # Collect the potential starting stations
#         stations = [i for i in range(len(gas)) if gas[i] >= cost[i]]

#         # Checking routes starting from each collected station
#         for i in stations:

#             station = i
#             tank = gas[i]

#             while tank >= 0:
                
#                 # Travel to the next station
#                 tank = tank - cost[station] 

#                 # Check if we actually can get to the next station with current gas
#                 if tank < 0:
#                     break
                    
#                 # If we are at the end of the stations (clockwise)
#                 if station + 1 == len(gas):
#                     station = 0
                            
#                 else:
#                     station += 1
                            
#                 #If we success in making the lap
#                 if station == i:
#                     return i
            
#                 # Refill the tank
#                 tank = tank + gas[station]

#         # in case no successful loop happens, return -1
#         return -1

#     # Testing
#     print(canCompleteCircuit(gas=gas, cost=cost))

#     'Note: My solution met 85% of the test cases'


#     'Another approach'
#     def canCompleteCircuit(gas:list[int], cost:list[int]) -> int:
        
#         # Handle the corner case
#         if sum(gas) < sum(cost):
#             return -1
        
#         current_gas = 0
#         starting_index = 0

#         for i in range(len(gas)):

#             current_gas += gas[i] - cost[i]

#             if current_gas < 0:
#                 current_gas = 0
#                 starting_index = i + 1
                
#         return starting_index
    
#     # Testing
#     print(canCompleteCircuit(gas=gas, cost=cost))

#     'Note: This simplified version prooved to be more efficient'

'''179. Largest Number'''
# def x():

#     # Input
#     # Case 1
#     nums = [20,1]
#     # Output: "201"

#     # Case 2
#     nums = [3,30,34,5,9]
#     # Output: "9534330"

#     # Custom Case
#     nums = [8308,8308,830]
#     # Output: "83088308830"


#     'My 1st Approach'
#     def largestNumber(nums: list[int]) -> str: 

#         nums = [str(x) for x in nums]
    
#         res = sorted(nums, key= lambda x: int(x[0]), reverse=True)  


#         # Mergesort
#         def mergesort(seq: list) -> list:

#             if len(seq) <= 1:
#                 return seq

#             mid = len(seq)//2

#             left_side, right_side = seq[:mid], seq[mid:]

#             left_side = mergesort(left_side)
#             right_side = mergesort(right_side)

#             return merge(left=left_side, right=right_side)

#         # Auxiliary merge for Mergesort
#         def merge(left: list, right: list) -> list:

#             res = []
#             zeros = []
#             i = j = 0

#             while i < len(left) and j < len(right):

#                 if left[i][-1] == '0':
#                     zeros.append(left[i])
#                     i+=1

#                 elif right[j][-1] == '0':
#                     zeros.append(right[j])
#                     j+=1
                
#                 elif left[i][0] == right[j][0]:

#                     if left[i]+right[j] > right[j]+left[i]:
#                         res.append(left[i])
#                         i+=1

#                     else:
#                         res.append(right[j])
#                         j+=1                

#                 elif int(left[i][0]) > int(right[j][0]):
#                     res.append(left[i])
#                     i+=1
                
#                 else:
#                     res.append(right[j])
#                     j+=1
            

#             while i < len(left):
#                 res.append(left[i])
#                 i+=1

            
#             while j < len(right):
#                 res.append(right[j])
#                 j+=1


#             # Deal with the elements with '0' as last digit
#             zeros.sort(key=lambda x: int(x), reverse=True)

#             return res+zeros          

#         result = mergesort(seq=res)
        
#         return ''.join(result)

#     # Testing
#     print(largestNumber(nums=nums))

#     'Note: This approach cleared 57% of cases '


#     'My 2nd Approach'
#     def largestNumber(nums: list[int]) -> str: 

#         res = [str(x) for x in nums]    
#         # res = sorted(nums, key= lambda x: int(x[0]), reverse=True)  

#         # Mergesort
#         def mergesort(seq: list) -> list:

#             if len(seq) <= 1:
#                 return seq

#             mid = len(seq)//2

#             left_side, right_side = seq[:mid], seq[mid:]

#             left_side = mergesort(left_side)
#             right_side = mergesort(right_side)

#             return merge(left=left_side, right=right_side)

#         # Auxiliary merge for Mergesort
#         def merge(left: list, right: list) -> list:

#             res = []        
#             i = j = 0

#             while i < len(left) and j < len(right):

#                 if left[i]+right[j] > right[j]+left[i]:
#                     res.append(left[i])
#                     i += 1

#                 else:
#                     res.append(right[j])
#                     j += 1
            
#             while i < len(left):
#                 res.append(left[i])
#                 i += 1
                            
#             while j < len(right):
#                 res.append(right[j])
#                 j += 1

#             return res        

#         result = mergesort(seq=res)
        
#         return ''.join(result)

#     # Testing
#     print(largestNumber(nums=nums))

#     'Note: This one did it!'

'''215. Kth Largest Element in an Array'''
# def x():

#     'Solution'
#     import heapq

#     def findKthLargest(self, nums: list[int], k: int) -> int:
#             heap = nums[:k]
#             heapq.heapify(heap)
            
#             for num in nums[k:]:
#                 if num > heap[0]:
#                     heapq.heappop(heap)
#                     heapq.heappush(heap, num)
            
#             return heap[0]

#     'Done'

'''218. The Skyline Problem'''
# def x():

#     '''
#     Explanation of the Code

#         Events Creation:

#             For each building, two events are created: entering ((left, -height, right)) and exiting ((right, height, 0)).
        
#         Sorting Events:

#             Events are sorted first by x-coordinate. If x-coordinates are the same, entering events are processed before exiting events. For entering events with the same x-coordinate, taller buildings are processed first.
        
#         Processing Events:

#             A max-heap (live_heap) keeps track of the current active buildings' heights. Heights are stored as negative values to use Python's min-heap as a max-heap.
#             When processing each event, heights are added to or removed from the heap as needed.
#             If the maximum height changes (top of the heap), a key point is added to the result.
        
#         This approach efficiently manages the skyline problem by leveraging sorting and a max-heap to dynamically track the highest building at each critical point.
#     '''

#     from heapq import heappush, heappop, heapify

#     def getSkyline(buildings: list[list[int]]) -> list[list[int]]:
            
#         # Create events for entering and exiting each building
#         events = []

#         for left, right, height in buildings:
#             events.append((left, -height, right))  # Entering event
#             events.append((right, height, 0))     # Exiting event
        

#         # Sort events: primarily by x coordinate, then by height
#         events.sort()
        

#         # Max-heap to store the current active buildings
#         result = []
#         live_heap = [(0, float('inf'))]  # (height, end)


#         # Process each event
#         for x, h, r in events:

#             if h < 0:  # Entering event
#                 heappush(live_heap, (h, r))

#             else:  # Exiting event
                
#                 # Remove the building height from the heap
#                 for i in range(len(live_heap)):
#                     if live_heap[i][1] == x:
#                         live_heap[i] = live_heap[-1]  # Replace with last element
#                         live_heap.pop()  # Remove last element
#                         heapify(live_heap)  # Restore heap property
#                         break
            
#             # Ensure the heap is valid
#             while live_heap[0][1] <= x:
#                 heappop(live_heap)
            
#             # Get the current maximum height
#             max_height = -live_heap[0][0]
            
#             # If the current maximum height changes, add the key point
#             if not result or result[-1][1] != max_height:
#                 result.append([x, max_height])
                    
#         return result

#     'Done'

'''334. Increasing Triplet Subsequence'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,3,4,5]
#     # Output: True / Any triplet where i < j < k is valid.

#     # Case 2
#     nums = [5,4,3,2,1]
#     # Output: False / Any triplet where i < j < k is valid.

#     # Case 3
#     nums = [2,1,5,0,4,6]
#     # Output: True / The triplet (3, 4, 5) where [0,4,6] is valid.

#     # Custom Case
#     nums = [1,2,2147483647]
#     # Output: False.


#     '''
#     My approach (Brute forcing) - Iterative looping

#         Intuition:

#             - Handle corner cases: 
#                 + If no input; 
#                 + if input length < 3; 
#                 + If input length = 3 != to sorted(input, reverse = False)
#                 + If input == sorted(input, reverse = True)

#             - In a while loop check one by one, starting from the first index, if next to it is any other element greater than it.
#                 from that element start the search for a greater element than the first greater and 
                
#                 + if found, return True;
#                 + else, move the initial index to the next and start over
#                 + if the initial index gets to the second last element and no triplet has been found, return False.
#     '''

#     def increasingTriplet(nums: list[int]) -> bool:

#         # Handle corner cases
#         if not nums or len(nums) < 3 or (len(nums) == 3 and nums != sorted(nums, reverse=True)) or nums == sorted(nums, reverse=True):
#             return False

#         # Initialize the triplet initial index
#         i = 0

#         # Iterate through the input elements
#         while i < len(nums)-2:

#             for j in range(i+1, len(nums)):

#                 if nums[j] > nums[i]:

#                     for k in range(j+1, len(nums)):

#                         if nums[k] > nums[j]:

#                             return True
                        
#             i += 1
        
#         return False

#     # Testing
#     print(increasingTriplet(nums=nums))

#     'Note: This approach met 90% of test cases, but failed with larger inputs. Time complexity: O(n^3)'


#     '''
#     My approach - Iterative selection

#         Intuition:

#             - Starting from the first index, check with listcomp if there is a larger element present.
#                 + if it does, get its index and do the same but for this second element.
#                     * if there are a larger element present return True,
#                     * else, move the initial input to the next and start over.

#             - Like the prior approach if it reaches the second last element in the input, end the loop and return False
#     '''

#     def increasingTriplet(nums: list[int]) -> bool:

#         # Handle corner cases
#         # if not nums or len(nums) < 3 or (len(nums) == 3 and nums != sorted(nums, reverse=True)) or nums == sorted(nums, reverse=True):
#         #     return False

#         # Initialize the triplet initial index
#         i = 0

#         # Iterate through the input elements
#         while i < len(nums)-2:

#             # Get the next greater element of nums[i]
#             sec_greater = list(filter(lambda x: x>nums[i], nums[i+1:-1]))

#             # if such element exist
#             if sec_greater:    
                
#                 # Iterate again for the rest of the greater elements
#                 for elem in sec_greater:

#                     # Get the idx of the first greater element than nums[i]
#                     j = nums.index(elem, i+1)            

#                     # Find a element greater than nums[j]
#                     third_greater = list(filter(lambda x: x>nums[j], nums[j+1:]))

#                     # if there are greater element than nums[j], return True
#                     if third_greater:
#                         return True       
                            
#             i += 1
        
#         return False

#     # Testing
#     print(increasingTriplet(nums=nums))

#     'Note: This approach met 90% of test cases, but failed with larger inputs. Time complexity: O(n^2*logn)'


#     'Optimized solution O(n)'

#     def increasingTriplet(nums: list[int]) -> bool:

#         first = float('inf')
#         second = float('inf')
        
#         for num in nums:

#             if num <= first:
#                 first = num

#             elif num <= second:
#                 second = num

#             else:
#                 return True
        
#         return False

#     # Testing
#     print(increasingTriplet(nums=nums))

#     'Done'





'''45. Jump Game II'''
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     nums = [2,3,1,1,4]
#     # Output: 2

#     # Case 2
#     nums = [2,3,0,1,4]
#     # Output: 2

#     # Custom case
#     nums = [1,2,1,1,1]
#     # Output: 2


#     '''
#     My Approach (DP Approach)
#     '''

#     def jump(nums: list[int]) -> int:
        
#         # Capture the input length
#         n = len(nums)

#         # Initialize the DP array
#         dp = [float('inf')] * n

#         # Assign 1 to the first place of dp, since is given that at least must be 1 jump to reach any other place
#         dp[0] = 0

#         # Process the input
#         for i in range(n-1):
            
#             # # Algorithm's early exit: If the first jump is long enough to reach the input's end
#             # if i+nums[i] >= n-1:
#             #     return dp[i]+1
            
#             for j in range(1, nums[i]+1):                
#                 dp[i+j] = min(dp[i+j], dp[i]+1)

#         # Return the dp's last position
#         return dp[-1] 

#     # # Testing
#     print(jump(nums=nums))

#     '''Note: This approach worked, beating submissions in time by 18% and space by 14%'''




#     '''
#     Greedy Approach

#         Explanation
            
#             - Variables:

#                 *jumps: This keeps track of how many jumps you make.
#                 *farthest: The farthest index you can reach from the current position.
#                 current_end: The boundary of the current jump, i.e., how far you can go with the current number of jumps.
            
#             - Logic:

#                 * You iterate through the list and update farthest to track the farthest position you can reach from any position within the current range.
#                 * Whenever you reach current_end, it means you must make a jump, so you increase the jumps counter and set current_end to farthest.
#                 * You stop if current_end reaches or exceeds the last index because you know you can jump directly to the end.
#     '''  

#     def jump(nums: list[int]) -> bool:

#         # Handle corner case: single element input
#         if len(nums) == 1:
#             return 0
        
#         # Initialize variables
#         jumps = 0  # Number of jumps
#         farthest = 0  # The farthest point that can be reached
#         current_end = 0  # The farthest point within the current jump range
        
#         # Traverse the array, but we don't need to check the last element
#         for i in range(len(nums) - 1):

#             # Update the farthest point that can be reached
#             farthest = max(farthest, i + nums[i])
            
#             # If we reach the end of the current jump's range, we must make a new jump
#             if i == current_end:

#                 jumps += 1
#                 current_end = farthest  # Update the range of the next jump
                
#                 # If the farthest we can reach is the last index or beyond, we can stop
#                 if current_end >= len(nums) - 1:
#                     break
        
#         return jumps

        
#     # Testing
#     print(jump(nums = nums))


#     '''Notes: Done'''

'''12. Integer to Roman'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     num = 3749
#     # Output: MMMDCCXLIX

#     # Case 2
#     num = 58
#     # Output: LVIII

#     # Case 3
#     num = 1994
#     # Output: MCMXCIV
    
#     # # Custom Case
#     # num = 0
#     # # Output: -

#     '''
#     My Approach

#         Intuition:
            
#             - Parse the input by digit in reverse.
#             - Assign a roman number according to if its units, tens, hundred or thousands
#             - Revert the order of the result built elements
#             - Join and return
#     '''

#     def intToRoman(num: int) -> str:

#         # Parse the input to separate by units, tens, hundreds and thousands but reversed
#         places = [int(x) for x in str(num)[::-1]]

#         # Initialize a Result Holder
#         res: list = []

#         # Process the input
#         for i in range(len(places)):

#             if i == 3:
#                 res.append('M'*places[i])
            
#             else:                                
#                 if places[i] != 0:

#                     num = places[i]*(10**i)
#                     first_dig = places[i]

#                     if first_dig in range(1,4):                    
#                         if i == 0:
#                             res.append('I'*first_dig)

#                         elif i == 1:
#                             res.append('X'*first_dig)
                        
#                         elif i == 2:
#                             res.append('C'*first_dig)


#                     elif first_dig == 4:
#                         if i == 0:
#                             res.append('IV')

#                         elif i == 1:
#                             res.append('XL')
                        
#                         elif i == 2:
#                             res.append('CD')


#                     elif first_dig == 5:
#                         if i == 0:
#                             res.append('V')

#                         elif i == 1:
#                             res.append('L')
                        
#                         elif i == 2:
#                             res.append('D')
                    

#                     elif first_dig in range(6,9):
#                         if i == 0:
#                             res.append('V'+'I'*(first_dig-5))

#                         elif i == 1:
#                             res.append('L'+'X'*(first_dig-5))
                        
#                         elif i == 2:
#                             res.append('D'+'C'*(first_dig-5))


#                     else:
#                         if i == 0:
#                             res.append('IX')

#                         elif i == 1:
#                             res.append('XC')
                        
#                         elif i == 2:
#                             res.append('CM')

#         # Reverse back and join the 'res' holder
#         res = ''.join(res[::-1])
        
#         return res


#     # Testing
#     print(intToRoman(num=num))

#     '''Note: While this approach works, it a bit verbose and could be confusing compared to the Greedy approach'''




#     '''
#     Greedy Approach
    
#         Explanation:

#         Roman Numeral Mapping:
#             Use a list of tuples to map integer values to their corresponding Roman numeral symbols. The list is ordered from largest to smallest.
        
#         Iterate Through the Map:
#             For each (value, symbol) in the map, repeatedly subtract value from num and append symbol to the result string until num is smaller than value.
        
#         Return the Result:
#             Once all values have been processed, the accumulated result contains the Roman numeral representation.
#     '''

#     def intToRoman(num: int) -> str:

#         # Define a list of tuples mapping Roman numeral values to their symbols
#         roman_map = [
#             (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
#             (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
#             (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I')
#         ]
        
#         # Initialize the result
#         result = ""
        
#         # Process the integer
#         for value, symbol in roman_map:

#             # Append the Roman numeral symbol while the value fits into num
#             while num >= value:
#                 result += symbol
#                 num -= value
        
#         return result
    

#     # Testing
#     print(intToRoman(num=num))

#     '''Note: Done'''











