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
189. Rotate Array (Array) (TP)
202. Happy Number (Hash Table) (TP) (Others)
234. Palindrome Linked List (LL) (RC) (TP)
283. Move Zeroes (Array) (TP)
287. Find the Duplicate Number (FCD) (Array) (TP)
344. Reverse String (TP)
350. Intersection of Two Arrays II (Array) (TP)

31. Next Permutation (Array) (TP)
142. Linked List Cycle II (Hash Table) (LL) (TP) (FCD)
27. Remove Element (Array) (TP)
392. Is Subsequence (TP) (DP)


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


(21)
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


#     'My Approach'
#     def merge(nums1, m, nums2, n):

#         i, j = 0, 0
#         res = []

#         while i < m and j < n:

#             if nums1[i] < nums2[j]:            
#                 res.append(nums1[i])
#                 i+=1
            
#             else:
#                 res.append(nums2[j])
#                 j += 1
        
#         while i < m:        
#             res.append(nums1[i])
#             i+=1

#         while j < n:        
#             res.append(nums2[j])
#             j+=1


#         for i in range(len(res)):
#             nums1[i] = res[i]

#     # Testing
#     merge(nums1=nums1, nums2=nums2, m=m, n=n)
#     print(nums1)

#     'Notes: This solution works but holds more memory than I know it could be, so lets try'


#     'Optimized Approach (Two-Pointers)'
#     def merge(nums1, nums2, m, n) -> None:

#         p1, p2, p = m-1, n-1, m+n-1
        
#         while p1 >= 0 and p2 >= 0:

#             if nums1[p1] > nums2[p2]:            
#                 nums1[p] = nums1[p1]
#                 p1 -= 1
            
#             else:
#                 nums1[p] = nums2[p2]
#                 p2 -= 1
            
#             p -= 1
    
#         while p2 >= 0:
#             nums1[p] = nums2[p2]
#             p2 -= 1
#             p -= 1 

#     # Testing
#     merge(nums1=nums1, nums2=nums2, m=m, n=n)
#     print(nums1)

#     'Notes: There it is!'

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

'''189. Rotate Array'''
# def x():

#     'Input'
#     # Case 1
#     nums, k = [1,2,3,4,5,6,7], 3
#     # Output: [5,6,7,1,2,3,4]

#     # Case 2
#     nums, k = [-1,-100,3,99], 2
#     # Output: [3,99,-1,-100]

#     # My approach
#     def rotate(nums: list[int], k: int) -> None:

#         if len(nums) == 1:
#             return
        
#         rot = k % len(nums)

#         dic = {k:v for k, v in enumerate(nums)}

#         for i in range(len(nums)):

#             n_idx = (i+rot)%len(nums)
#             nums[n_idx] = dic[i]

#     'Note:It actually worked!'

'''202. Happy Number'''
# def x():

#     # Input
#     # Case 1
#     n = 19
#     # Output: True

#     # Case 2
#     n = 2
#     # Output: False

#     # Custom Case
#     n = 18
#     # Output: False


#     '''
#     My Approach
    
#         Intuition (Recursive)
            
#             - Recursively separate the digits and check the sum of their squares compared to 1.
#                 - If the stackoverflow is reached, return False.            
#     '''

#     def isHappy(n:int) -> bool:

#         def aux(m:int) -> bool:

#             num = [int(x)**2 for x in str(m)]
#             num = sum(num)

#             if num == 1:
#                 return True
            
#             return aux(m=num)
        
#         try:
#             res = aux(m=n)

#             if res:
#                 return True
        
#         except RecursionError as e:        
#             return False

#     # Testing
#     print(isHappy(n=n))

#     'This approach may work but it exceed time limit: only met 4% of cases'


#     '''
#     Set Approach

#     There are mainly two ways of solving this: The set approach and the Floyd's Cycle detection algorithm

#         - The set approach: Use a set to save the seen numbers and if you end up in one of them, you entered a cycle
#         - The Floyd's Cycle Detection Algorithm: Similar to the case of catching a cycle in a linked list with two pointers: Slow and Fast.
#     '''

#     'Set Approach'
#     def isHappy(n:int) -> bool:

#         def getNum(m:int)->int:
#             return sum(int(x)**2 for x in str(m))

#         seen = set()

#         while n != 1 and n not in seen:
#             seen.add(n)
#             n = getNum(n)
        
#         return n == 1

#     # Testing
#     print(isHappy(n=n))


#     'FDC Approach'
#     def isHappy(n:int) -> bool:

#         def getNum(m:int)->int:
#             return sum(int(x)**2 for x in str(m))

#         slow = n
#         fast = getNum(n)

#         while fast != 1 and slow != fast:
#             slow = getNum(slow)
#             fast = getNum(getNum(fast))
        
#         return fast == 1

#     # Testing
#     print(isHappy(n=n))

#     'Done'

'''234. Palindrome Linked List'''
# def x():

#     # Base
#     # Definition for singly-linked list.
#     class ListNode:
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input
#     # Case 1
#     head_layout = [1,2,2,1]
#     head = ListNode(val=1, next=ListNode(val=2, next=ListNode(val=2, next=ListNode(val=1))))
#     # Output: True

#     # Case 2
#     head_layout = [1,2]
#     head = ListNode(val=1, next=ListNode(val=2))
#     # Output: False

#     # Custom Case
#     head_layout = [1,0,0]
#     head = ListNode(val=1, next=ListNode(val=0, next=ListNode(val=0)))
#     # Output: False


#     '''
#     My Approach (Brute forcing)

#         Intuition:
#             - Traverse the list collecting the values.
#             - Return the test that the values collected are equal to their reverse.
#     '''

#     def is_palindrome(head:ListNode) -> bool:
        
#         # Define values holder
#         visited = []    

#         # Traverse the list
#         while head:
            
#             visited.append(head.val)

#             head = head.next

#         return visited == visited[::-1]

#     # Testing
#     print(is_palindrome(head=head))

#     '''Note: 
#         This is the most "direct" way to solve it, but there are two more way to solve this same challenge
#         One involves recursion/backtracking and the other solve the problem with O(1) of space complexity, while this and
#         The recursive approaches consumes O(n).'''


#     '''
#     Recursive Approach

#         Intuition:
#             - Make a pointer to the head of the llist (will be used later).
#             - Define the Auxiliary recursive function:
#                 + This function will go in depth through the list and when it hits the end,
#                     it will start to go back in the call stack (which is virtually traversing the list backwards).
#                 + When the reverse traversing starts compare each node with the pointer defined at the begining and if they have equal values
#                     it means up to that point the palindrome property exist, otherwise, return False.
#                 + If the loop finishes, it means the whole list is palindromic.
#             - return True.
#     '''
#     class Solution:

#         def __init__(self) -> None:
#             pass

#         def is_palindrome(self, head:ListNode) -> bool:

#             self.front_pointer = head

#             def rec_traverse(current_node:ListNode) -> bool:

#                 if current_node is not None:
                    
#                     if not rec_traverse(current_node.next):
#                         return False
                    
#                     if self.front_pointer.val != current_node.val:
#                         return False
                
#                     self.front_pointer = self.front_pointer.next

#                 return True
            
#             return rec_traverse(head)
    
#     # Testing
#     solution = Solution()
#     print(solution.is_palindrome(head=head))

#     'Note: The solution as a -standalone function- is more complex than as a class method'


#     '''
#     Iterative Approach / Memory-efficient

#         Intuition:
#             - Use a two-pointer approach to get to the middle of the list.
#             - Reverse the next half (from the 'slow' pointer) of the llist.
#             - Initiate a new pointer to the actual head of the llist and in a loop (while 'the prev node')
#                 compare the two pointer up until they are different or the 'prev' node gets to None.
#             - If the loop finishes without breaking, return 'True'.
#     '''

#     def is_palindrome(head:ListNode) -> bool:

#         # Hanlde corner cases:
#         if not head or not head.next:
#             return True
        

#         # Traverse up to the middle of the llist
#         slow = fast = head

#         while fast and fast.next:
#             slow = slow.next
#             fast = fast.next.next

        
#         # Reverse the remaining half of the llist
#         prev = None

#         while slow:
#             next_node = slow.next
#             slow.next = prev
#             prev = slow
#             slow = next_node


#         # Compare the reversed half with the actual first half of the llist
#         left, right = head, prev

#         while right:

#             if left.val != right.val:
#                 return False
            
#             left, right = left.next, right.next

        
#         # If it didn't early end then means the llist is palindromic
#         return True

#     # Testing
#     print(is_palindrome(head=head))
    
#     'Done'

'''283. Move Zeroes'''
# def x():

#     # Input
#     # Case 1
#     nums = [0,1,0,3,12]
#     # Output: [1,3,12,0,0]

#     # Case 2
#     nums = [0]
#     # Output: [0]

#     # Custom Case
#     nums = [2,3,4,0,5,6,8,0,1,0,0,0,9]
#     # Output: [0]


#     '''
#     My Approach

#         Intuition:
#             - Create a new list as a buffer to hold every item in the initial order
#             - Separate the buffer into non-zeroes and zeroes different list and joint them together.
#             - Replace each value of the original list with the order or the buffer list.

#         This solution is more memory expensive than one with a Two-pointer approach, but let's try it
#     '''

#     def move_zeroes(nums:list[int]) -> None:

#         # Handle corner case
#         if len(nums) == 1:
#             return nums
    
#         # Create the buffers to separate the non-zeroes to the zeroes
#         non_zeroes, zeroes = [x for x in nums if x != 0],[x for x in nums if x == 0]

#         # Join the buffers into one single list
#         buffer = non_zeroes + zeroes

#         # Modify the original input with the buffer's order
#         for i in range(len(nums)):
#             nums[i] = buffer[i]
    
#     # Testing
#     move_zeroes(nums=nums)
#     print(nums)

#     'Note: This solution was accepted and beated submissions by 37% in runtime and 87% in memory'


#     'Two-pointers Approach'
#     def move_zeroes(nums:list[int]) -> None:

#         # Initialize the left pointer
#         l = 0

#         # Iterate with the right pointer through the elements of nums
#         for r in range(len(nums)):

#             if nums[r] != 0:

#                 nums[r], nums[l] = nums[l], nums[r]

#                 l += 1

#     # Testing
#     move_zeroes(nums=nums)
#     print(nums)

#     'Done'

'''287. Find the Duplicate Number'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,3,4,2,2]
#     # Output: 2

#     # Case 2
#     nums = [3,1,3,4,2]
#     # Output: 3

#     # Custom Case
#     nums = [3,3,3,3,3]
#     # Output: 3

#     'My approach'

#     def find_duplicate(nums:list[int]) -> int:

#         for num in nums:

#             if nums.count(num) != 1:
#                 return num
    
#     # Testing
#     print(find_duplicate(nums=nums))

#     'Note: This approach cleared 92% of cases but breaks with larger inputs'


#     'Hare & Tortoise Approach'
#     def find_duplicate(nums:list[int]) -> int:

#         # Initialize two pointers directing to the first element in the list
#         slow = fast = nums[0]

#         # Iterate until they coincide (They' found each other in the cycle)
#         while True:
#             slow = nums[slow]
#             fast = nums[nums[fast]]
            
#             if slow == fast:
#                 break
        
#         # Reset the slow to the begining of the list, so they an meet at the repeating number
#         slow = nums[0]

#         # Iterate again but at same pace, they will eventually meet at the repeated number
#         while slow != fast:
#             slow = nums[slow]
#             fast = nums[fast]

#         return fast

#     # Testing
#     print(find_duplicate(nums=nums))

#     'Done'

'''344. Reverse String'''
# def x():

#     'Note: The problem ask for modify in place an iterable'

#     'Two pointers approach'
#     def reverse_string(s: list[str]) -> None:

#         left, right = 0, len(s)

#         while left < right:

#             s[left], s[right] = s[right], s[left]

#             left += 1
#             right -= 1

#     'Done'

'''350. Intersection of Two Arrays II'''
# def x():

#     # Input
#     # Case 1
#     nums1, nums2 = [1,2,2,1], [2,2]
#     # Output: [2,2]

#     # Case 2
#     nums1, nums2 = [4,9,5], [9,4,9,8,4]
#     # Output: [4,9]


#     '''
#     My approach

#         Intuition:
#             - Handle a corner case.
#             - Make a list holder for the result.
#             - Get the largest list.
#             - Collect the common elements and populate the result holder with the lower count from both inputs.
#     '''

#     def intersect(nums1: list[int], nums2: list[int]) -> list[int]:

#         # Handle corner case
#         if not nums1 or not nums2:
#             return []

#         # Create a list holder for the common elements
#         commons = []

#         # Create an iterator with the longest list
#         longest = nums1 if len(nums1)>len(nums2) else nums2
        
#         # Collect the common elements
#         for elem in longest:

#             if elem in nums1 and elem in nums2 and elem not in commons:

#                count = nums1.count(elem) if nums1.count(elem) < nums2.count(elem) else nums2.count(elem)

#                commons.extend([elem]*count)
        
#         return commons

#     # Testing
#     print(intersect(nums1=nums1, nums2=nums2))

#     'Note: This approach worked and beated only 5% in runtine and 93% in memory'


#     'Two pointer approach'
#     def intersect(nums1: list[int], nums2: list[int]) -> list[int]:

#         # Sort both arrays
#         nums1.sort()
#         nums2.sort()
        
#         # Initialize pointers and the result list
#         i, j = 0, 0
#         result = []
        
#         # Traverse both arrays
#         while i < len(nums1) and j < len(nums2):

#             if nums1[i] < nums2[j]:
#                 i += 1

#             elif nums1[i] > nums2[j]:
#                 j += 1

#             else:
#                 result.append(nums1[i])
#                 i += 1
#                 j += 1
        
#         return result

#     # Testing
#     print(intersect(nums1=nums1, nums2=nums2))

#     'Done'





'''31. Next Permutation'''
# def x():

#     from typing import List

#     # Input
#     # Case 1
#     nums = [1,2,3]
#     # Output: [1,3,2]

#     # Case 2
#     nums = [3,2,1]
#     # Output: [1,2,3]

#     # Case 3
#     nums = [1,1,5]
#     # Output: [1,5,1]

#     # Custom Case
#     nums = [6,7,5,3,5,6,2,9,1,2,7,0,9]
#     # Output: [5,1,1]


#     '''
#     My Approach (Brute forcing)

#         Intuition:
            
#             - The permutation position in the result of the iterable from the permutations generated from Permutation function from itertools module.
#             - Find the next permutation. 
#                 if the permutation is the last one, get the begining one with modulo operator applied to the index retrived.
#             - Modify the actual 'nums' and return.
#     '''

#     def nextPermutation(nums: List[int]) -> None:

#         # Handle Corner case: 1-element input
#         if len(nums) == 1:
#             return 
        
#         # Import itertools 'permutation
#         from itertools import permutations

#         # Capute the current nums as a tuple
#         curr = tuple(nums)

#         # Sort lexicographically the input
#         nums.sort()

#         # Generate a list of the lexicographically ordered permutations of nums
#         perm = sorted(set(permutations(nums)))

#         # Capture the index of the next permutation 'nums' in curr variable
#         idx = (perm.index(curr) + 1) % len(perm) # Is bounded at 'len(perm)' to get the 1st perm in case the current perm is the last one

#         # Get the next permutation called from the permutations list and the idx
#         next_perm = perm[idx]

#         # Modify the current permutation (nums) with the items of the next one
#         for i in range(len(nums)):
#             nums[i] = next_perm[i]


#     # Testing
#     print(f'''Initial permutation: {nums}''')

#     nextPermutation(nums=nums)
    
#     print(f'''Next permutation: {nums}''')

#     '''
#     Notes: 
#         This approach works but exceeded time limit at 24% of test cases. Since it need to generate all
#          the permutarions of an iterable to continue and finish, its time complexity is O(n!) factorial (making it
#          practically unfeasible)

#         The actual O(n) solutions is based on a 'cleaver observation' of how permutations behave and a two pointers approach.          
#     '''


#     '''Optmized Solution'''

#     def nextPermutation(nums: List[int]) -> None:

#         # Step 1: Find the first decreasing element
#         n = len(nums)
#         i = n - 2

#         while i >= 0 and nums[i] >= nums[i + 1]:
#             i -= 1
        
#         if i >= 0:  # If there is a valid i (the array isn't entirely in descending order)

#             # Step 2: Find the next larger element in the suffix
#             j = n - 1

#             while nums[j] <= nums[i]:
#                 j -= 1

#             # Swap the elements
#             nums[i], nums[j] = nums[j], nums[i]
        
#         # Step 3: Reverse the suffix starting from i + 1
#         left, right = i + 1, n - 1
#         while left < right:
#             nums[left], nums[right] = nums[right], nums[left]
#             left += 1
#             right -= 1

#     '''
#     Explanation

#         Key Insight: Finding the First Decreasing Element

#             Why? When looking for the "next permutation", the challenge is essentially asking: "How can we slightly increase the current permutation, so that we get the next smallest possible one?"

#             * If you look at a permutation, the part at the end is typically in descending order when you've reached the last possible permutation for that segment.
#             * For example, in [2, 3, 1], the last part [3, 1] is in descending order.
#             * In lexicographical order, permutations tend to increase at the last possible point. So, we need to find where to make that increase.
            
#             Step 1: Find the first decreasing element

#                 - Starting from the end, scan the list from right to left to find the first index i where nums[i] < nums[i + 1].
#                 - This is the point where we can "increase" the permutation. Everything after this point is already the largest possible permutation of that suffix.
#                 - Example: For [1, 3, 2], you find that nums[0] = 1 is less than nums[1] = 3, so i = 0. We want to increase the value at i.

#             Step 2: Finding the Next Larger Element

#                 Now that we’ve identified the point (i) where we can "increase" the permutation, the next step is to find the smallest possible number to swap with nums[i] to form the next permutation.

#                 Why? We want to change the permutation minimally, so we need to find the smallest number in the suffix that is greater than nums[i].

#                 Step 2: Find the next larger element in the suffix:

#                     - Scan the right part of the list (after i) to find the smallest element that is larger than nums[i].
#                     - This is because we want to "increase" the permutation as little as possible.
#                     - Example: In [1, 3, 2], nums[0] = 1, and the smallest element larger than 1 is 2 (at index 2), so we swap nums[0] with nums[2].
                
#                 After the swap, the list becomes [2, 3, 1]. Notice how this is larger than [1, 3, 2] but still close to it.

#             Step 3: Reverse the Suffix
            
#                 Now, the reason for this part is subtle but important.

#                 Why reverse the suffix?

#                 * After swapping nums[i] with the next larger element, the portion of the list after i is still in descending order. This descending part was previously the largest possible permutation of those numbers, but now that we’ve made a swap, we need to reset it to the smallest possible permutation.
#                 * To get the smallest possible permutation of the suffix, we just need to reverse it. This gives us the next lexicographically smallest order after the swap.
                
#                 Example:

#                     - After swapping, we have [2, 3, 1]. The part after i = 0 is [3, 1], which is in descending order.
#                     - To get the next permutation, we reverse [3, 1] to make it [1, 3].
#                     - The final result is [2, 1, 3], which is the next permutation after [1, 3, 2] in lexicographical order.

#     '''

'''142. Linked List Cycle II'''
# def x():
    
#     from typing import Optional

#     # Definition for singly-linked list.
#     class ListNode:
#         def __init__(self, x):
#             self.val = x
#             self.next = None

#     # Input
#     # Case 1
#     head_layout = [3,2,0,-4]
#     head = ListNode(3)
#     two = ListNode(2)
#     three = ListNode(0)
#     four = ListNode(-4)
#     head.next, two.next, three.next, four.next = two, three, four, two
#     # Output: Node in position 1

#     # # Case 2
#     # head_layout = [1,2]
#     # head = ListNode(1)
#     # two = ListNode(2)
#     # head.next, two.next = two, head
#     # # Output: Node in position 0
    
#     # Custom Case
#     head_layout = [1,2]
#     head = ListNode(1)
#     two = ListNode(2)
#     head.next, two.next = None, None
#     # Output: Node in position 0

#     '''
#     My Approach (Hash Table Approach)

#         Intuition:
            
#             - Create a Hash Table that stores each node while traversing the list as key and the node pointed by its 
#                 'next' attribute as its value.

#                 Check Membership of each value to be added, 
                    
#                     if it's already in the hash table, return that value as result.
#                     if the value is None, then return None as no cycle is in that list.
#     '''

#     def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:

#         # Create the Hash Table holder
#         nodes = {}

#         # Initialize a node in head to traverse the list
#         curr = head

#         while curr:

#             if curr.next in nodes:

#                 # If the node pointed by 'next' is already in the table, that's the begining of the cylce
#                 return curr.next
            
#             # Add the node to the holder
#             nodes[curr] = curr.next

#             # Move up to the next node
#             curr = curr.next
        
#         # If the loop finishes it means there was no cycle
#         return None

#     # Testing
#     print(detectCycle(head=head).val)


#     '''Note: This approach worked but it consumes O(n) memory, and is asked if could be done in O(1). Runtime: 12%; Memory: 8%'''




#     '''
#     My Approach (FCD/Tortoise and Hare Approach)

#         To reach the O(1) Memory
#         Intuition:
            
#             Build the Totroise and Hare cycle detection .
#     '''

#     def detectCycle(head: Optional[ListNode]) -> Optional[ListNode]:

#         # Initialize both pointers to the beginning of the list
#         slow, fast = head, head

#         # Traverse the LList
#         while fast and fast.next:

#             slow = slow.next
#             fast = fast.next.next

#             # If slow and fast meet, a cycle exists
#             if slow == fast:

#                 # Move slow pointer back to head and move both pointers one step at a time
#                 slow = head

#                 while slow != fast:
#                     slow = slow.next
#                     fast = fast.next

#                 # Return the node where they meet, i.e., the cycle's starting point
#                 return slow

#         # If the loop finishes without a meeting point, there is no cycle
#         return None

#     # Testing
#     print(detectCycle(head=head))

'''27. Remove Element'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     nums = [3,2,2,3]
#     val = 3
#     # Output: 2

#     # Case 2
#     nums = [0,1,2,2,3,0,4,2]
#     val = 2
#     # Output: 5

#     '''
#     My Approach (Two-Pointers)

#         The idea is to traverse while sorting the array as requested to achieve at least O(n) time complexity.

#         Intuition:
            
#             - Handle corner case: return 0 if len(nums) == 0.
#             - Handle corner case: return len(nums) if val not in nums.

#             - Create two pointers 'p1', 'p2', initilized in 0 and 1 respectively.
#             - In a While Loop (p2 < len(nums)):
#                 + if p1 and p2 are equal to val, move p2 to the right (p2 += 1).
#                 + elif p1 is equal to val, 
#                     * interchage the values nums[p1] and nums[p2]
#                     * move both pointer to the right (p1, += 1, p2 += 1)
#                 + else,
#                     * move p1 to the right (p1 += 1)
            
#             - Return the index in p1 that will be the total amount of elements different than val up to p1 if nums[p1] is val,
#                 Otherwise, return p1+1, since is a 0-based indexing and from 0 to p1 will be elements distinct to val.
#                     (return p1 + 1 if nums[p1] != val else p1)
#     '''

#     def removeElement(nums: list[int], val: int) -> int:

#         # Handle Corner case: len(nums) == 0
#         if len(nums) == 0:
#             return 0
        
#         # Handle Corner case: val not in nums
#         if val not in nums:
#             return len(nums)
        

#         # Create two pointers 'p1', 'p2'
#         p1, p2 = 0, 1

#         # Process nums
#         while p2 < len(nums):

#             if nums[p1] == val and nums[p2] == val:                
#                 p2 += 1
            
#             elif nums[p1] == val:
#                 nums[p1], nums[p2] = nums[p2], nums[p1]
#                 p1 += 1
#                 p2 += 1
            
#             elif nums[p1] != val and nums[p2] != val:                
#                 p1 += 1
#                 p2 += 1

#             else:
#                 p1 += 1        
        
#         # Return p1 or p1 + 1
#         return p1 + 1 if nums[p1] != val else p1

#     # Testing
#     print(removeElement(nums=nums, val=val))

#     '''Note: Done'''

'''392. Is Subsequence'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     s = "abc"
#     t = "ahbgdc"
#     # Output: true

#     # Case 2
#     s = "axc"
#     t = "ahbgdc"
#     # Output: false

#     '''
#     My Approach (Dynamic Programming)

#         Intuition:
            
#             - Here the same principle of the LCS algorithm is exactly the same but the returning statement it won't be
#                 the return the length of the longest common subsequence, but to compare that to the length of 's' which
#                 should be the same to say that s is a subsequence of t.
#     '''

#     def isSubsequence(s: str, t: str) -> bool:

#         # Handle Corner case: t being shorter tha s
#         if len(t) < len(s):
#             return False
        
#         # Handle Corner case: s is equal to t
#         if s == t:
#             return True
        
#         m, n = len(s), len(t)

#         dp = [[0]*(n+1) for _ in range(m+1)]

#         for i in range(1, m+1):
#             for j in range(1, n+1):

#                 if s[i-1] == t[j-1]: 
#                     dp[i][j] = dp[i-1][j-1] + 1
                
#                 else:
#                     dp[i][j] = max(dp[i][j-1], dp[i-1][j])
        
#         return len(s) == dp[-1][-1]


#     # Testing
#     print(isSubsequence(s=s, t=t))

#     '''Note: Done'''

    
#     '''
#     My Approach (Two Pointers)

#         Intuition:
            
#             - Simply iterate and move over both inputs.
#     '''

#     def isSubsequence(s: str, t: str) -> bool:

#         # Initilize both pointers to 0
#         p1, p2 = 0, 0

#         # Iterate through both inputs at same time
#         while p1 < len(s) and p2 < len(t):

#             if s[p1] == t[p2]:
#                 p1 += 1
            
#             p2 += 1       
        
#         return p1 == len(s)


#     # Testing
#     print(isSubsequence(s=s, t=t))

#     '''Note: Done'''






















