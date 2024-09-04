'''
CHALLENGES INDEX

11. Container With Most Water (Array) (TP) (GRE)
55. Jump Game (Array) (DP) (GRE)
122. Best Time to Buy and Sell Stock II (Array) (DP) (GRE)
134. Gas Station (Array) (GRE)
179. Largest Number (Array) (Sorting) (GRE)
215. Kth Largest Element in an Array (Array) (Heap) (DQ) (Sorting)
218. The Skyline Problem (Heaps) (DQ)


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













