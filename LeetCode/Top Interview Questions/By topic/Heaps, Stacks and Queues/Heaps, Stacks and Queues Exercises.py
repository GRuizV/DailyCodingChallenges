'''
CHALLENGES INDEX

20. Valid Parentheses (Stack)
23. Merge k Sorted Lists (LL) (DQ) (Heap) (Sorting)
42. Trapping Rain Water (Array) (TP) (DS) (Stack)


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
*Arrays, Hash Tables & Matrices
*Sorting
*Heaps, Stacks & Queues
*Graphs, Trees & Binary Trees

(XX)
'''



'20. Valid Parentheses'
# def x():

#     # input / Case - expected result
#     s = '()'    # True
#     s = '()[]{}'    # True
#     s = '(]'    # False
#     s = '([({[]{}}())])'    # True
#     s = '([({[)]{}}())])'    # False
#     s = '))'    # False
#     s = '(('    # False

#     # My approach
#     def isValid(s):

#         stack = list(s)
#         temp = []
#         dic = {'(': ')', '[':']', '{':'}'}  

#         while True:

#             if len(stack) == 0 and len(temp) != 0:
#                 return False

#             popped = stack.pop(-1)

#             if popped in '([{':
                
#                 if len(temp) == 0 or temp[0] != dic[popped]:
#                     return False
                                
#                 else:                
#                     temp = temp[1:]

#             else:
#                 temp.insert(0,popped)

#             if len(stack) == 0 and len(temp)==0:
#                 return True  

#     # Testing
#     print(isValid(s))

#     'Notes: it works!'

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


















