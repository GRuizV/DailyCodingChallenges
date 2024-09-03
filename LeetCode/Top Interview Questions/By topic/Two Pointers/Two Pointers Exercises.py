'''
CHALLENGES INDEX

5. Longest Palindromic Substring (DP) (TP)
11. Container With Most Water (Array) (TP) (GRE)
15. 3Sum (Array) (TP) (Sorting)
42. Trapping Rain Water (Array) (TP) (DS) (Stack)
75. Sort Colors (Array) (TP) (Sorting)
88. Merge Sorted Array (Array) (TP) (Sorting)
125. Valid Palindrome (TP)
141. Linked List Cycle (TP) (LL)
148. Sort List (TP) (LL)
160. Intersection of Two Linked Lists (TP) (LL)


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


'5. Longest Palindromic Substring'
# def x():

#     # Input
#     s = "cbbd"


#     # 1st Approach: Brute Force

#     # Creating the possible substrings from the input
#     subs = []

#     for i in range(1, len(s)+1):
        
#         for j in range((len(s)+1)-i):

#             subs.append(s[j:j+i])

#     # # validating
#     # print(subs)        

#     palindromes = sorted(filter(lambda x : True if x == x[::-1] else False, subs), key=len, reverse=True)

#     print(palindromes)

#     '''
#     Note: While the solution works, is evidently not efficient enough / Time Limit Exceeded.
#     '''

#     # 2nd Approach: Same brute force but less brute

#     max_len = 1
#     max_str = s[0]

#     for i in range(len(s)-1):

#         for j in range(i+1, len(s)):

#             sub = s[i:j+1]        

#             if (j-i)+1 > max_len and sub == sub[::-1]:

#                 max_len = (j-i)+1
#                 max_str = s[i:j+1]


#     print(max_str)

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

'42. Trapping Rain Water'
# def x():

#     # Input

#     # case 1
#     height = [0,1,0,2,1,0,1,3,2,1,2,1]  # Exp. Out: 6

#     # case 2
#     height = [4,2,0,3,2,5]  # Exp. Out: 9


#     'Solution'
#     def trap(height):

#         if not height:
#             return 0
        

#         left, right = 0, len(height)-1
#         left_max, right_max = 0, 0
#         result = 0

#         while left < right:

#             if height[left] < height[right]:

#                 if height[left] >= left_max:
#                     left_max = height[left]

#                 else:
#                     result += left_max - height[left]

#                 left += 1
            
#             else:

#                 if height[right] >= right_max:
#                     right_max = height[right]

#                 else:
#                     result += right_max - height[right]

#                 right -= 1
        
#         return result

#     # Testing
#     print(trap([3,0,2]))

#     'Done'

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

'''125. Valid Palindrome'''
# def x():
        
#     def isPalindrome(s:str) -> bool:
#         s = ''.join([x for x in s if x.isalpha()]).casefold()
#         return s == s[::-1]

#     # Testing
#     a = '0PO'
#     # a = ''.join([x for x in a if x.isalnum()]).casefold()
#     print(isPalindrome(a))

#     'Done'

'''141. Linked List Cycle'''
# def x():

#     # Base
#     class ListNode(object):
#         def __init__(self, x):
#             self.val = x
#             self.next = None

#     # Input
#     # Case 1
#     head_layout = [3,2,0,-4]
#     head = ListNode(x=3)
#     pos1 = ListNode(x=2)
#     pos2 = ListNode(x=0)
#     pos3 = ListNode(x=-4)
#     head.next, pos1.next, pos2.next, pos3.next = pos1, pos2, pos3, pos1
#     # Output: True / Pos1

#     # Case 2
#     head_layout = [1,2]
#     head = ListNode(x=1)
#     pos1 = ListNode(x=2)
#     head.next, pos1.next = pos1, head
#     # Output: True / Pos0

#     # Case 3
#     head_layout = [1]
#     head = ListNode(x=1)
#     # Output: False / pos-1

#     'My Approach'
#     def hasCycle(head:ListNode) -> bool:

#         # Hanlde Corner Case
#         if head is None or head.next == None:
#             return False
        

#         visited = []
#         curr = head

#         while curr is not None:

#             if curr in visited:
#                 return True
            
#             visited.append(curr)

#             curr = curr.next
        
#         return False

#     # Testing
#     print(hasCycle(head=head))

#     'Note: This a suboptimal solution, it works but it takes considerable memory to solve it'


#     '''
#     Another approach (Probing)

#     Explanation
        
#         By making two markers initialized in the head one with the double of the "speed" of the other, if those are in a cycle
#         at some point they got to meet, it means there is a cycle in the list, but if one if the faster gets to None,
#         that'll mean that there is no cycle in there.
#     '''

#     def hasCycle(head:ListNode) -> bool:

#         if not head:
#             return False
        
#         slow = fast = head

#         while fast and fast.next:

#             slow = slow.next
#             fast = fast.next.next

#             if slow == fast:
#                 return True
        
#         return False

#     # Testing
#     print(hasCycle(head=head))

#     'Done'

'''148. Sort List'''
# def x():

#     # Base
#     class ListNode(object):
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input
#     # Case 1
#     list_layout = [4,2,1,3]
#     head = ListNode(val=4, next=ListNode(val=2, next=ListNode(val=1, next=ListNode(val=3))))
#     # Output: [1,2,3,4]

#     # Case 2
#     list_layout = [-1,5,3,4,0]
#     head = ListNode(val=-1, next=ListNode(val=5, next=ListNode(val=3, next=ListNode(val=4, next=ListNode(val=0)))))
#     # Output: [-1,0,3,4,5]

#     # Case 3
#     list_layout = [1,2,3,4]
#     head = ListNode(val=1, next=ListNode(val=2, next=ListNode(val=3, next=ListNode(val=4))))
#     # Output: [1,2,3,4]


#     '''
#     My Approach
    
#         Intuition:

#             - Brute force: Traverse the list to collect each node with its value in a list,
#             and apply some sorting algorithm to sort them.
#     '''

#     def sortList(head):

#         if not head:
#             return ListNode()
        
#         curr = head
#         holder = []

#         while curr:

#             holder.append([curr.val, curr])
#             curr = curr.next


#         def merge_sort(li):

#             if len(li)<=1:
#                 return li
            
#             left_side = li[:len(li)//2]
#             right_side = li[len(li)//2:]

#             left_side = merge_sort(left_side)
#             right_side = merge_sort(right_side)

#             return merge(left=left_side, right=right_side)


#         def merge(left, right):
            
#             i = j = 0
#             result = []

#             while i < len(left) and j < len(right):

#                 if left[i][0] < right[j][0]:
#                     result.append(left[i])
#                     i+=1
                
#                 else:
#                     result.append(right[j])
#                     j+=1

#             while i < len(left):
#                 result.append(left[i])
#                 i+=1
            
#             while j < len(right):
#                 result.append(right[j])
#                 j+=1

#             return result

#         sorted_list = merge_sort(li=holder)
        
#         for i in range(len(sorted_list)):

#             if i == len(sorted_list)-1:
#                 sorted_list[i][1].next = None
            
#             else:
#                 sorted_list[i][1].next = sorted_list[i+1][1]
        
#         return sorted_list[0][1]

#     # Testing
#     test = sortList(head=head)

#     'Done'

'''160. Intersection of Two Linked Lists'''
# def x():

#     # Base
#     class ListNode(object):
#         def __init__(self, x):
#             self.val = x
#             self.next = None

#     # Input
#     # Case 1
#     listA, listB = [4,1,8,4,5], [5,6,1,8,4,5]
#     a1, a2 = ListNode(x=4), ListNode(x=1)
#     c1, c2, c3 = ListNode(x=8), ListNode(x=4), ListNode(x=5)
#     b1, b2, b3 = ListNode(x=5), ListNode(x=6), ListNode(x=1)
#     a1.next, a2.next = a2, c1
#     c1.next, c2.next = c2, c3
#     b1.next, b2.next, b3.next = b2, b3, c1
#     #Output: 8

#     # Case 2
#     listA, listB = [1,9,1,2,4], [3,2,4]
#     a1, a2, a3 = ListNode(x=1), ListNode(x=9), ListNode(x=1)
#     c1, c2 = ListNode(x=2), ListNode(x=4)
#     b1 = ListNode(x=3)
#     a1.next, a2.next, a3.next = a2, a3, c1
#     c1.next = c2
#     b1.next = c1
#     # Output: 2

#     # Case 3
#     listA, listB = [2,6,4], [1,5]
#     a1, a2, a3 = ListNode(x=2), ListNode(x=6), ListNode(x=4)
#     b1, b2 = ListNode(x=1), ListNode(x=5)
#     a1.next, a2.next = a2, a3
#     b1.next = b2
#     # Output: None


#     '''
#     My approach

#         Intuition
#             - Traverse the first list saving the nodes in a list
#             - Traverse the second list while checking if the current node is in the list
#                 - If so, return that node
#                 - Else, let the loop end
#             - If the code gets to the end of the second loop, means there isn't a intersection.
#     '''

#     def getIntersectionNode(headA = ListNode, headB = ListNode) -> ListNode:

#         visited_nodes = []

#         curr = headA

#         while curr:
#             visited_nodes.append(curr)
#             curr = curr.next

#         curr = headB

#         while curr:
            
#             if curr in visited_nodes:
#                 return curr
            
#             curr = curr.next
            
#         return None

#     # Testing
#     result = getIntersectionNode(headA=a1, headB=b1)
#     print(result.val) if result else print(None)

#     'Note: This solution breaks when the data input is too large in leetcode, it got up to 92% of cases'


#     'Two pointers Approach'
#     def getIntersectionNode(headA = ListNode, headB = ListNode) -> ListNode:

#         a, b = headA, headB

#         while a != b:
        
#             if not a:
#                 a = headB

#             else:
#                 a = a.next
            
#             if not b:
#                 b = headA
            
#             else:
#                 b = b.next
        
#         return a

#     # Testing
#     result = getIntersectionNode(headA=a1, headB=b1)
#     print(result.val) if result else print(None)

#     '''
#     Explanation

#         The logic here is that with two pointer, each one directed to the head of each list,
#         if both exhaust their lists and star with the other, if there are intersected they MUST
#         meet at the intersection node after traversing both lists respectviely or otherwise they will be 'None'
#         at same time after the second lap of the respective lists.
# '''























