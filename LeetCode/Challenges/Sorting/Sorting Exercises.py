'''
CHALLENGES INDEX

15. 3Sum (Array) (TP) (Sorting)
23. Merge k Sorted Lists (LL) (DQ) (Heap) (Sorting)
49. Group Anagrams (Array) (Hash Table) (Sorting)
56. Merge Intervals (Array) (Sorting)
75. Sort Colors (Array) (TP) (Sorting)
88. Merge Sorted Array (Array) (TP) (Sorting)
179. Largest Number (Array) (Sorting) (GRE)
215. Kth Largest Element in an Array (Array) (Heap) (DQ) (Sorting)
295. Find Median from Data Stream (Heap) (Sorting)
347. Top K Frequent Elements (Array) (Heaps) (Sorting)


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


'15. 3Sum'
# def x():

#     import itertools

#     # Input
#     nums = [0,0,0]


#     # My approach

#     '''
#     Rationale:
        
#         1) Build all combinations caring for the order.
#         2) Filter down those who met sum(subset) = 0
#         3) Make sure there is no duplicates & return.

#     '''
#     comb = list(itertools.combinations(nums,3))

#     comb = [sorted(x) for x in comb if sum(x) == 0]

#     res = []

#     for i in comb:

#         if i not in res:
#             res.append(i)

#     print(res)

#     '''
#     Notes:

#         This solution actually works, but breaks when a big enough input is passed.
#     '''

#     # Two-Pointers approach solution
#     def threeSum(self, nums):
            
#             nums.sort()
#             answer = []
            
#             # if the inputs have less than 3 items
#             if len(nums) < 3:
#                 return answer
            
#             for i in range(len(nums)):

#                 # Since is a sorted input, if first element is positive, there is no way it'll sum up to 0
#                 if nums[i] > 0:
#                     break
                
#                 # Apart from the first element, if the following is the same, jump to the next iteration to avoid returning duplicates
#                 if i > 0 and nums[i] == nums[i - 1]:
#                     continue
                
#                 # Pointers setting    
#                 low, high = i + 1, len(nums) - 1

#                 while low < high:

#                     s = nums[i] + nums[low] + nums[high]

#                     if s > 0:
#                         high -= 1

#                     elif s < 0:
#                         low += 1

#                     else:

#                         answer.append([nums[i], nums[low], nums[high]])
#                         lastLowOccurrence, lastHighOccurrence = nums[low], nums[high]
                        
#                         while low < high and nums[low] == lastLowOccurrence:
#                             low += 1
                        
#                         while low < high and nums[high] == lastHighOccurrence:
#                             high -= 1
            
#             return answer

'23. Merge k Sorted Lists'
# def x():

#     # Base
#     class ListNode(object):
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input

#     # 1st Input
#     #List 1
#     one1, two1, three1 = ListNode(1), ListNode(4), ListNode(5)
#     one1.next, two1.next = two1, three1

#     #List 2
#     one2, two2, three2 = ListNode(1), ListNode(3), ListNode(4)
#     one2.next, two2.next = two2, three2

#     #List 3
#     one3, two3 = ListNode(2), ListNode(6)
#     one3.next = two3

#     # List of lists
#     li = [one1, one2, one3]

#     # My Approach

#     '''
#     Rationale:
    
#         1. Create an empty node.
#         2. Assign the node with the minimum value as next
#         3. Move that node to its next node until reaches 'None'.
#         4. When every value within the input list is None, breakout the loop and return.
#     '''

#     def mergeKLists(lists:list[ListNode]) -> ListNode:
        
#         lists = [x for x in lists if x.val != '']

#         if len(lists) == 0:
#             return ListNode('')


#         head = ListNode('')
#         curr = head
#         li = lists

#         while True:

#             if li == [None]:
#                 break

#             # Create a list of the current nodes in input that aren't None and sort them ascendingly by value
#             li = sorted([node for node in li if node != None], key = lambda x: x.val)

#             # Make the 'next_node' the next node to the curr None & move over to that node right away
#             curr.next = li[0]
#             curr = curr.next

#             # Move over to the next node of next_node
#             li[0] = li[0].next

#         return head.next

#     # Testing
#     res = mergeKLists([ListNode('')])
#     res_li = []

#     print(res)

#     'Notes: It worked'

'49. Group Anagrams'
# def x():

#     # Input

#     # Case 1
#     strs = ["eat","tea","tan","ate","nat","bat"]
#     #Exp. Out: [["bat"],["nat","tan"],["ate","eat","tea"]]

#     # Case 2
#     strs = [""]
#     #Exp. Out: [[""]]

#     # Case 3
#     strs = ["a"]
#     # Exp. Out: [["a"]]

#     # Custom Case
#     strs = ["ddddddddddg","dgggggggggg"]
#     # Expected: [["dgggggggggg"],["ddddddddddg"]]



#     'My Approach'

#     '''
#     Intuition:
#         1. Take the first element of the input and make a list with all element that contains the same characters
#         2. Erase the taken elements from the input.
#         3. Reiterate steps 1 & 2 until the input is exhausted

#     '''

#     def groupAnagrams(strs:list):
        
#         if len(strs) == 1:
#             return[strs]

#         # Auxiliary anagram checker
#         def is_anagram(ref:list, string:list):

#             if len(ref) != len(string):
#                 return False

#             for char in ref:
                
#                 if ref.count(char) != string.count(char):   
#                     return False

#             return True
        
#         # Creating Flag to manage repetitions
#         strs = [[word, False] for word in strs]


#         result = []

#         for word in strs:
                
#             if word[1] == False:

#                 anagrams = []
#                 anagrams.append(word[0])            
#                 word[1] = True

#                 for rest in strs:

#                     if rest[1] == False:

#                         if is_anagram(word[0], rest[0]):
#                             anagrams.append(rest[0])
#                             rest[1] = True
            
#                 result.append(anagrams)

#         return result
    
#     # Testing
#     print(groupAnagrams(strs))

#     '''
#     Notes: 
#         It passed 72/126 cases, the case below broke the code: 
#             strs = ["ddddddddddg","dgggggggggg"] / Output: [["ddddddddddg","dgggggggggg"]], Expected: [["dgggggggggg"],["ddddddddddg"]]

#         After the fixture, it works but beat no one in efficiency
#     '''


#     'Another Approach'
#     def groupAnagrams(strs):
        
#         freq = {}

#         for word in strs:

#             newWord = ''.join(sorted(word))

#             if newWord not in freq:
#                 freq[newWord] = []
            
#             freq[newWord].append(word)

#         return list(freq.values())

#     # Testing
#     print(groupAnagrams(strs))

#     '''
#     Notes: Absolutely more elegant solution
#     '''

'56. Merge Intervals'
# def x():

#     #Input
#     # Case 1
#     intervals = [[1,3],[2,6],[8,10],[15,18]]
#     # Output: [[1,6],[8,10],[15,18]]

#     # Case 2
#     intervals = [[1,4],[4,5]]
#     # Output: [[1,5]]

#     # Custom Case
#     intervals = [[1,4],[0,0]]
#     # Output: [...]

#     '''
#     Intuition:

#         - Check the second item of the element and the first of the next, 
#         if they coincide, merge.
#             (Through a While Loop)
#     '''

#     def merge(intervals:list[list[int]]) -> list[list[int]]:

#         #Handling the corner case
#         if len(intervals) == 1:
#             return intervals

#         intervals.sort(key=lambda x: x[0])

#         idx = 0

#         while idx < len(intervals)-1:

#             if intervals[idx][1] >= intervals[idx+1][0]:

#                 merged_interval = [[min(intervals[idx][0], intervals[idx+1][0]), max(intervals[idx][1], intervals[idx+1][1])]]
#                 intervals = intervals[:idx] + merged_interval + intervals[idx+2:]
#                 idx = 0

#             else:
#                 idx += 1

#         return intervals

#     # Testing
#     print(merge(intervals))

#     'Note: My solution works but is not efficient, since it has to go over the whole array again'

    
#     'Some other Approach'
#     def merge(intervals):

#         intervals.sort()

#         merge_intervals = []
#         curr_interval = intervals[0]

#         for interval in intervals[1:]:

#             if curr_interval[1] < interval[0]:
#                 merge_intervals.append(curr_interval)
#                 curr_interval = interval

#             else:
#                 curr_interval[1] = max(curr_interval[1], interval[1])

#         merge_intervals.append(curr_interval)

#         return merge_intervals

#     # Testing
#     print(merge(intervals))

'75. Sort Colors'
# def x():

#     # Input
#     # Case 1
#     nums = [2,0,2,1,1,0]
#     # Output: [0,0,1,1,2,2]

#     # Case 2
#     nums = [2,0,1]
#     # Output: [0,1,2]

#     '''
#     My approach

#         Intuition:
#             Since the solution requires the sorting be in place, 
#             perhaps Bubblesort would do the trick.
#     '''

#     def sortColors(nums:list[int]) -> list[int]:

#         swapped = True       

#         while swapped != False:

#             swapped = False
#             i = 0

#             while True:

#                 if i == len(nums)-1:
#                     break

#                 if nums[i] > nums[i+1]:
#                     nums[i], nums[i+1] = nums[i+1], nums[i]
#                     swapped = True

#                 i += 1

#     # Testing
#     sortColors(nums)
#     print(nums)

#     'Notes: Done!'

'88. Merge Sorted Array'
# def x():

#     # Input
#     # Case 1
#     nums1 = [1,2,3,0,0,0]
#     m = 3
#     nums2 = [2,5,6]
#     n = 3
#     # Output: [1,2,2,3,5,6]

#     # Case 2
#     nums1 = [1]
#     m = 1
#     nums2 = []
#     n = 0
#     # Output: [1]

#     # Case 3
#     nums1 = [0]
#     m = 0
#     nums2 = [1]
#     n = 1
#     # Output: [1]

#     # Custom case
#     nums1 = [0,2,0,0,0,0,0]
#     m = 2
#     nums2 = [-1,-1,2,5,6]
#     n = 5
#     # Output: [1]

#     # Custom case
#     nums1 = [-1,1,0,0,0,0,0,0]
#     m = 2
#     nums2 = [-1,0,1,1,2,3]
#     n = 6
#     # Output: [1]


#     'Solution'
#     def merge(nums1, m, nums2, n):

#         if m == 0:
#             for i in range(n):
#                 nums1[i] = nums2[i]

#         elif n != 0:

#             m = n = 0

#             while n < len(nums2):

#                 if nums2[n] < nums1[m]:

#                     nums1[:m], nums1[m+1:] = nums1[:m] + [nums2[n]], nums1[m:-1]

#                     n += 1
#                     m += 1
                
#                 else:

#                     if all([x==0 for x in nums1[m:]]):
#                         nums1[m] = nums2[n]
#                         n += 1
                        
#                     m += 1

#     # Testing
#     merge(nums1,m,nums2,n)
#     print(nums1)

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

'''295. Find Median from Data Stream'''
# def x():

#     # Input
#     # Case 1
#     commands = ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
#     inputs = [[], 1, 2, [], 3, []]
#     # Output: [None, None, None, 1.5, None, 2.0]

#     # Case 2
#     commands = ["MedianFinder","addNum","findMedian","addNum","findMedian","addNum","findMedian","addNum","findMedian","addNum","findMedian"]
#     inputs = [[],-1,[],-2,[],-3,[],-4,[],-5,[]]
#     # Output: [None, None, -1.0, None, -1.5, None, -2.0, None, -2.5, None, -3.0]


#     'My approach'
#     import heapq

#     class MedianFinder:

#         def __init__(self):
#             self.nums = []      

#         def addNum(self, num: int) -> None:

#             # heapq.heappush(self.nums, num)
#             # heapq.heapify(self.nums)

#             self.nums.append(num)
#             self.nums.sort()
            

#         def findMedian(self) -> float:

#             nums_len = len(self.nums)

#             if nums_len % 2 == 0:
#                 mid1, mid2 = nums_len//2-1, nums_len//2
#                 return (self.nums[mid1]+self.nums[mid2])/2
            
#             else:            
#                 mid = nums_len//2
#                 return self.nums[mid]

#     # Testing
#     obj = MedianFinder()

#     for idx, command in enumerate(commands):

#         if command == 'addNum':
#             print(obj.addNum(inputs[idx]))
        
#         elif command == 'findMedian':
#             print(obj.findMedian())


#     'Note: This solution met 95% of cases but sorting in every addition cauases inefficiencies'


#     'Heaps approach'
#     import heapq

#     class MedianFinder:

#         def __init__(self):
#             self.small = []  # Max-heap (inverted values)
#             self.large = []  # Min-heap

#         def addNum(self, num: int) -> None:

#             # Add to max-heap (invert to simulate max-heap)
#             heapq.heappush(self.small, -num)
            
#             # Balance the heaps
#             if self.small and self.large and (-self.small[0] > self.large[0]):
#                 heapq.heappush(self.large, -heapq.heappop(self.small))
            
#             # Ensure the sizes of the heaps differ by at most 1
#             if len(self.small) > len(self.large) + 1:
#                 heapq.heappush(self.large, -heapq.heappop(self.small))

#             if len(self.large) > len(self.small):
#                 heapq.heappush(self.small, -heapq.heappop(self.large))

#         def findMedian(self) -> float:

#             # If the heaps are of equal size, median is the average of the tops
#             if len(self.small) == len(self.large):
#                 return (-self.small[0] + self.large[0]) / 2
            
#             # Otherwise, the median is the top of the max-heap
#             return -self.small[0]

#     'Done'

'''347. Top K Frequent Elements'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,2,1]
#     k = 2
#     # Output: [1,2]

#     # Case 2
#     nums = [1]
#     k = 1
#     # Output: [1]


#     '''
#     My approach

#         Intuition:
            
#         Ideas' pool:
#             + A Counter function approach:
#                 - Call a Counter on the input and sort by freq, return in order.

#             + A Heap approach:
#                 - ...    
#     '''

#     def topKFrequent(nums: list[int], k: int) -> list[int]:

#         # Create the result list holder
#         result = []

#         # Import Counter
#         from collections import Counter

#         #  Transform the input into a list sorted by freq
#         nums = sorted(Counter(nums).items(), key=lambda x: x[1], reverse=True)

#         # Populate the result accordingly
#         for i in range(k):
#             result.append(nums[i][0])

#         # Return the result
#         return result

#     # Testing
#     print(topKFrequent(nums=nums, k=k))

#     'Note: This approach worked beating submissions only 20% in Runtime and 61% in Memory'

#     'Done'














