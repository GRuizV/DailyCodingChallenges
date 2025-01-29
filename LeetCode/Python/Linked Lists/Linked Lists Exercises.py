'''
CHALLENGES INDEX

2. Add Two Numbers (LL)
19. Remove Nth Node From End of List (LL) (TP)
21. Merge Two Sorted Lists (LL) (RC)
23. Merge k Sorted Lists (LL) (DQ) (Heap) (Sorting)
138. Copy List with Random Pointer (Hash Table) (LL)
141. Linked List Cycle (TP) (LL)
148. Sort List (TP) (LL)
160. Intersection of Two Linked Lists (TP) (LL)
206. Reverse Linked List (LL) (RC)
234. Palindrome Linked List (LL) (RC) (TP)
237. Delete Node in a Linked List (LL)
328. Odd Even Linked List (LL)

24. Swap Nodes in Pairs (LL)
25. Reverse Nodes in k-Group (LL)
114. Flatten Binary Tree to Linked List (LL) (DFS) (Tree)
142. Linked List Cycle II (Hash Table) (LL) (TP) (FCD)


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


(16)
'''


class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None



'2. Add Two Numbers'
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     l1 = [2,4,3]
#     l2 = [5,6,4]
#     l1 = ListNode(2)
#     l1.next = ListNode(4)
#     l1.next.next = ListNode(3)
#     l2 = ListNode(5)
#     l2.next = ListNode(6)
#     l2.next.next = ListNode(4)
#     # Output: [7,0,8]

#     # Case 2
#     l1 = [0]
#     l2 = [0]
#     l1 = ListNode(0)
#     l2 = ListNode(0)
#     # Output: [0]

#     # Case 3
#     l1 = [9,9,9,9,9,9,9]
#     l2 = [9,9,9,9]
#     l1 = ListNode(9)
#     l1.next = ListNode(9)
#     l1.next.next = ListNode(9)
#     l1.next.next.next = ListNode(9)
#     l1.next.next.next.next = ListNode(9)
#     l1.next.next.next.next.next = ListNode(9)
#     l1.next.next.next.next.next.next = ListNode(9)
#     l2 = ListNode(9)
#     l2.next = ListNode(9)
#     l2.next.next = ListNode(9)
#     l2.next.next.next = ListNode(9)
#     # Output: [8,9,9,9,0,0,0,1]

#     # Case 4
#     l1 = [2,4,9]
#     l2 = [5,6,4,9]
#     l1 = ListNode(2)
#     l1.next = ListNode(4)
#     l1.next.next = ListNode(9)
#     l2 = ListNode(5)
#     l2.next = ListNode(6)
#     l2.next.next = ListNode(4)
#     l2.next.next.next = ListNode(9)
#     # Output: [7,0,4,0,1]


#     '''
#     My Approach

#         Intuition:
            
#             - Initialize an empty node called 'head'.
#             - Initialize another empty node called 'current'.
#             - Assign point 'head' next to 'current'.
#             - Initialize a holder 'carry' to hold the remainder of the sums of the elements at 0.
#             - in a while loop (while l1 and l2):
                
#                 + if l1:
#                     * 'carry' += l1
#                     * l1 = l1.next

#                 + if l2:
#                     * 'carry' += l2
#                     * l2 = l2.next

#                 + if 'carry' > 9:
#                     * Assign the last digit of 'carry' as 'current' value pointer.
#                     * redefine 'carry' as:
#                         - 'carry' -= current.val
#                         - 'carry' /= 10

#                 + else:
#                     * Assign 'carry' to 'current' value pointer.
#                     * reset 'carry' to 0.

#                 + Assign 'current' next pointer to an empty node.
#                 + Assign 'current' to 'current' 'next' pointer.

            
#             - Return head's next pointer

#     '''

#     def addTwoNumbers(l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:

#         # Initialize an empty node called 'head'
#         head = ListNode(0)

#         # Initialize a node holder at 'head'.
#         current = head
       
#         # Initialize a holder 'carry'
#         carry = 0

#         # Process the input
#         while l1 or l2 or carry:
            
#             value1 = l1.val if l1 else 0
#             value2 = l2.val if l2 else 0

#             total = value1 + value2 + carry

#             carry = total // 10
#             current.next = ListNode(total%10)

#             current = current.next

#             if l1:
#                 l1 = l1.next
            
#             if l2:
#                 l2 = l2.next       

#         # Return the head's next 
#         return head.next

#     # Testing
#     head = addTwoNumbers(l1=l1, l2=l2)
#     result = []

#     while head:
#         result.append(head.val)
#         head=head.next
    
#     print(result)

#     '''Note: Done'''

'19. Remove Nth Node From End of List'
# def x():

#     # Base
#     class ListNode(object):
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next



#     # input
#     # [1, 2, 3, 4, 5]
#     one, two, three, four, five = ListNode(1), ListNode(2), ListNode(3), ListNode(4), ListNode(5), 
#     one.next, two.next, three.next, four.next = two, three, four, five

#     # # input 2
#     #     # [1, 2]
#     one, two = ListNode(1), ListNode(2)
#     one.next = two


#     # My Approach
#     def removeNthFromEnd(head:ListNode, n:int) -> ListNode:
       
#         # Handiling the special case [1]
#         if not head.next:
#             return ListNode('')

#         # Setting variables
#         curr = head
#         next_node = None
#         list_len = 0

#         # Probing the list and defining the target        
#         while curr:   
#             list_len += 1
#             curr = curr.next   

#         target = list_len - n

#         curr = head # Resetting the current node to modify the list

#         # Handiling the special case tagert = 0
#         if target == 0:
#             return curr.next
        
#         # Getting to the node before the target
#         for _ in range(1, target):
#             curr = curr.next

#         # Actually modifying the list
#         curr.next = curr.next.next

#         return head

#     # Testing
#     result = removeNthFromEnd(None, one, 2)


#     # Verifying the functioning
#     li = []

#     # if one.next is None:
#     #     print(result)

#     if True:
#         while result is not None:
#             li.append(result.val)
#             result = result.next

#         print(li)
    
#     'Notes: It works!'

'21. Merge Two Sorted Lists'
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     list1 = [1,2,4]
#     list2 = [1,3,4]
#     list1 = ListNode(1)
#     list1.next = ListNode(2)
#     list1.next.next = ListNode(4)
#     list2 = ListNode(1)
#     list2.next = ListNode(3)
#     list2.next.next = ListNode(4)
#     # Output: [1,1,2,3,4,4]
    

#     '''
#     My Approach

#         Intuition:
            
#             - Handle corner case: No list1 nor list2, return None.
#             - Handle corner case: No list1 or list2, return the other list.
#             - Create an empty node 'dummy'.
#             - Initialize a node holder 'current' at 'dummy'.
            
#             - In a while loop (while list1 and list2):
                
#                 + if list1 and list2:

#                     * if list1.val > list2.val:
#                         - current.next = list1
#                         - list1 = list1.next
                    
#                     * else:
#                         - current.next = list2
#                         - list2 = list2.next
                
#                 + if not list2:
#                     * current.next = list1
                                    
#                 + if not list1:
#                     * current.next = list2

#                 + current = current.next
            
#             - Return dummy.next

#     '''

#     def mergeTwoLists(list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:

#         # Handle Corner case:  No list1 nor list2
#         if not list1 and not list2:
#             return None
        
#         # Handle corner case: No list1 or list2, return the other list.
#         if not list1:
#             return list2

#         if not list2:
#             return list1

#         # Create an empty node
#         dummy = ListNode(0)

#         # Initialize a node holder 'current' at 'dummy'
#         current = dummy


#         while list1 and list2:

#             if list1 and list2:

#                 if list1.val < list2.val:
#                     current.next = list1
#                     list1 = list1.next
                
#                 else:
#                     current.next = list2
#                     list2 = list2.next
            
#             current = current.next


#         if list1 and not list2:
#             current.next = list1
            

#         if list2 and not list1:
#             current.next = list2
           
        
#         # Return dummy's next pointer
#         return dummy.next


#     # Testing
#     head = mergeTwoLists(list1=list1, list2=list2)
#     res = []
#     while head:
#         res.append(head.val)
#         head = head.next
#     print(res)

#     '''Note: Done'''

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

'''138. Copy List with Random Pointer'''
# def x():

#     from typing import Optional

#     # Definition for a Node.
#     class Node:
#         def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
#             self.val = int(x)
#             self.next = next
#             self.random = random

#     # Input
#     # Case 1
#     head = [[7,None],[13,0],[11,4],[10,2],[1,0]]
#     head = Node(7, next=Node(13, next=Node(11, next=Node(10, next=Node(1)))))
#     head.random = None
#     head.next.random = head
#     head.next.next.random = head.next.next.next.next
#     head.next.next.next.random = head.next.next
#     head.next.next.next.next.random = head
#     # Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]

#     # Case 2
#     head = [[1,1],[2,1]]
#     head = Node(1, next=Node(2))
#     head.random = head.next
#     head.next.random = head.next
#     # Output: [[1,1],[2,1]]

#     # Case 3
#     head = [[3,None],[3,0],[3,None]]
#     head = Node(3, next=Node(3, next=Node(3)))
#     head.random = None
#     head.next.random = head
#     head.next.next.random = None
#     # Output: [[3,None],[3,0],[3,None]]

#     # Case 4
#     head = []
#     head = None

#     # Case 5
#     head = [[1,1]]
#     head = Node(1)
#     head.random = head
#     # Output: [[1,1]]


#     '''
#     My Approach

#         Intuition:
            
#             - Handle corner case: No node passed.
#             - Initialize a 'dummy' node at 'head'.
#             - Create an empty dictionary named 'dic'.
#             - In a While Loop (while dummy):
#                 + Add a new entry to the dict:
#                     * With 'dummy' as its key
#                     * And a new node with dummy.val as 'val'.
#                 + Move 'dummy' to its next pointer.

#             - Reset back dummy to head.
#             - In the same Loop (while dummy):
#                 + to the value in dic[dummy] in its 'next' pointer set it as the content
#                     in dic[dummy.next] if dic[dummy.next] exists otherwise set it to None.
#                 + to the value in dic[dummy] in its 'random' pointer set it as the content
#                     in dic[dummy.random] if dic[dummy.random] exists otherwise set it to None. 
            
#             - Return dic[head]'s content.
#     '''

#     def copyRandomList(head: Optional[Node]) -> Optional[Node]:

#         # Handle Corner case: No node passed
#         if not head:
#             return None

#         # Initialize a 'dummy' node at 'head
#         dummy = head

#         # Create an empty dictionary
#         dic = {}

#         # Build the dic's content
#         while dummy:

#             dic[dummy] = Node(dummy.val)
#             dummy = dummy.next
        
#         # Reset back dummy to head
#         dummy = head

#         # Populate each of dic's content pointers
#         while dummy:
#             new_node = dic[dummy]
#             new_node.next = dic.get(dummy.next)
#             new_node.random = dic.get(dummy.random)
#             dummy = dummy.next
        
        
#         # Return the dic's in 'head's content
#         return dic[head]

#     # Testing
#     dummy = copyRandomList(head=head)
#     res = []

#     while dummy:
#         res.append([dummy.val, dummy.random.val if dummy.random else None])
#         dummy = dummy.next

#     print(res)

#     '''Note: Done'''

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

'''206. Reverse Linked List'''
# def x():

#     # Base 
#     # Definition for singly-linked list.
#     class ListNode:
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input
#     # Case 1
#     head_layout = [1,2,3,4,5]
#     head = ListNode(val=1)
#     two, three, four, five = ListNode(2), ListNode(3), ListNode(4), ListNode(5),
#     head.next, two.next, three.next, four.next = two, three, four, five
#     # Output: [5,4,3,2,1]

#     # Case 2
#     head_layout = [1,2]
#     head, two, = ListNode(1), ListNode(2)
#     head.next = two
#     # Output: [2,1]

#     # Case 3
#     head_layout = []
#     head = None
#     # Output: []


#     'Solution'
#     def reverseList(head:ListNode) -> ListNode:
        
#         # Initialize node holders
#         prev = None
#         curr = head    

#         while curr:
#             next_node = curr.next
#             curr.next = prev
#             prev = curr
#             curr = next_node       
        
#         return prev


#     def rec_reverseList(head:ListNode) -> ListNode:
        
#         # Base case
#         if not head or not head.next:
#             return head   

#         # Recursive Call
#         new_head = rec_reverseList(head.next)

#         # Reversing the list
#         head.next.next = head
#         head.next = None

#         return new_head

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

'''237. Delete Node in a Linked List'''
# def x():

#     # Definition for singly-linked list.
#     class ListNode:
#         def __init__(self, x):
#             self.val = x
#             self.next = None

#     # Input
#     # Case 1
#     llist = [4,5,1,9]
#     node = ListNode(5)
#     head = ListNode(4)
#     head.next = node
#     node.next = ListNode(1)
#     node.next.next = ListNode(9)
#     # Output: [4,1,9]

#     # Case 2
#     llist = [4,5,1,9]
#     node = ListNode(1)
#     head = ListNode(4)
#     head.next = ListNode(5)
#     head.next.next = node
#     node.next = ListNode(9)
#     # Output: [4,5,9]


#     '''
#     Solution

#         Intuition:
#             - The only way to modify the list in place without accessing the head of the list is to overwrite
#                 the value of the given node with the next, and when reach the end, point the last node to None.
#     '''

#     def delete_node(node:ListNode) -> None:
#         node.val = node.next.val
#         node.next = node.next.next

#     #Testing
#     delete_node(node=node)
#     new_node = head

#     while new_node:
#         print(new_node.val, end=' ')
#         new_node = new_node.next

#     'Done'

'''328. Odd Even Linked List'''
# def x():

#     from typing import Optional

#     # Base
#     # Definition for singly-linked list.
#     class ListNode:
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input
#     # Case 1
#     list_map = [1,2,3,4,5]
#     head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
#     # Output: [1,3,5,2,4]

#     # Case 2
#     list_map = [2,1,3,5,6,4]
#     head = ListNode(2, ListNode(1, ListNode(3, ListNode(5, ListNode(6, ListNode(4))))))
#     # Output: [2,3,6,1,5,4]


#     '''
#     My Approach

#         Intuition:
#             - Create a mock node (even) to hold its respective members.
#             - Traverse the list and each even node conect it to the mock node and the odd node conected to the next of its consequent even.
#             - Conect the tail of the original node (now the odd nodes) to the mocking node 'even'.
#     '''

#     def oddEvenList(head: Optional[ListNode]) -> Optional[ListNode]:

#         # Handle corner case: There has to be a node to do something.
#         if head:        

#             # Initialize the 'even' nodes holder
#             even = ListNode()

#             # Traverse the LList & separete odds from even
#             curr = head
#             curr_even = even
            
#             while curr:

#                 if curr.next:
                    
#                     # Assign the connection
#                     curr_even.next = curr.next
#                     curr.next = curr.next.next

#                     # Continue traversing
#                     curr = curr.next
#                     curr_even = curr_even.next
                
#                 else:
#                     break


#             # if the lenght of the LList is odd:
#             #   The tail of odds is a node and must be connect to the even head, and the tail of even must point to None.
#             # if is even:
#             #   The tail of odd is None and the list mus be traversed again to connect its tail to even's head and the tail of even is already pointing to None.
            
#             if curr:
#                 curr_even.next = None
#                 curr.next = even.next

#             else:
#                 curr = head

#                 while curr.next:
#                     curr = curr.next
                
#                 curr.next = even.next
            
#             # As the modification was in place, there is no return statement

#     # Testing
#     oddEvenList(head=head)
#     curr = head

#     while curr:
#         print(curr.val, end=' ')
#         curr = curr.next

#     'Note: This approached worked, it beated 71% of submissions in Runtime and 38% in Memory'


#     'A cleaner approach of the same'
#     def oddEvenList(head):
#         if not head or not head.next:
#             return head

#         odd = head
#         even = head.next
#         even_head = even

#         while even and even.next:
#             odd.next = even.next
#             odd = odd.next
#             even.next = odd.next
#             even = even.next

#         odd.next = even_head

#         return head

#     'Done'




'''24. Swap Nodes in Pairs'''
# def x():

#     from typing import Optional

#     # Base
#     # Definition for singly-linked list.
#     class ListNode:
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input
#     # Case 1
#     head = ListNode(1, next=ListNode(2,next=ListNode(3,next=ListNode(4))))
#     # Output: [2,1,4,3]

#     '''
#     Solution

#         Explanation:
            
#             The key here is not the swapping itself but the notion of having a node before the 'first' and 'second' nodes to
#             swap, since that node is the one that will connect back to 'second' and reconnecting the list.

#             This happens at first in curr = dummy, that is a dummy node with value 0 that goes before the actual LL, that way
#             is possible to make the swap with 'first' connecting to the node right next to 'second' and connecting 'second' to 'first',
#             the crucial part is just after that happens, when curr connects back to 'second', that way the list makes sense again.
#     '''

#     def swapPairs(head: Optional[ListNode]) -> Optional[ListNode]:

#         # Handle Corner case: no input.
#         if not head:
#             return
        
#         # Handle Corner case: 1-element LL.
#         if not head.next:
#             return head
        
#         # Create a dummy node to ease the head operations
#         dummy = ListNode(0, next=head)

#         # Initialize the current pointer
#         curr = dummy

#         # Start altering the list while traversing it
#         while curr.next and curr.next.next:

#             # Nodes to be swapped
#             first = curr.next
#             second = curr.next.next

#             # Swapping the nodes
#             first.next = second.next
#             second.next = first
#             curr.next = second

#             # Move the current pointer two nodes ahead
#             curr = first
        
#         # Return the new head
#         return dummy.next

#     # Testing
#     print(swapPairs(head=head))

#     '''Note: Done'''

'''25. Reverse Nodes in k-Group'''
# def x():

#     from typing import Optional

#     # Base
#     # Definition for singly-linked list.
#     class ListNode:
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input
#     # Case 1
#     head = ListNode(1, next=ListNode(2,next=ListNode(3,next=ListNode(4, next=ListNode(5)))))
#     k = 2
#     # Output: [2,1,4,3,5]

#     # Case 2
#     head = ListNode(1, next=ListNode(2,next=ListNode(3,next=ListNode(4, next=ListNode(5)))))
#     k = 3
#     # Output: [3,2,1,4,5]


#     '''
#     My Approach

#         Intuition:

#             - Handle Corner Cases
#             - Define an auxiliary LList reverse function.  
#             - Capture the head of the list in a 'new_head' node.  

#             within a while True loop
#             - Traverse the nodes from the 'new_head' node up to k position and count how many node there are.
#                 - Break case: if current is None.
#             - Capture the node in position k (1-indexed) and the node next to it as 'next_node'.
#             - Point the kth node to None.
#             - Redefine the 'new_head' node as the result of passing 'new_head' node to the aux func.
#             - Traverse again the list from 'new_head' to k and point the kth node to the 'next_node'.
#             - Redefine the 'new_head' to be the 'next_node' node.

#             - Return the head node.

#     '''

#     # Aux reversal function
#     def reverse_ll(head: Optional[ListNode]) -> Optional[ListNode]:

#         # Define a dummy node initilized in 'None'
#         prev = None

#         # Capture the 'head' node and the node next to it
#         curr = head
        
#         # Iterate thru the llist while 'curr' node is not none
#         while curr:
            
#             # Capture the node next to the current
#             next_node = curr.next

#             # Point the curr node 'next' pointer to the dummy
#             curr.next = prev

#             # Modify the dummy to be the current 'curr' node
#             prev = curr

#             # Modify 'curr' to be the current 'next_node' node
#             curr = next_node

#         # Return the 'dummy' since is the new 'head' of the llist
#         return prev
    
#     # Actual solution
#     def reverseKGroup(head: Optional[ListNode], k: int) -> Optional[ListNode]:

#         # Handle Corner case: If an empty LL is passed, return None
#         if not head:
#             return
        
#         # Handle Corner case: If k=1, then no reverse takes place
#         if k==1:
#             return head

#         # Create a dummy node to ease head operations
#         dummy = ListNode(0)
#         dummy.next = head

#         # The 'Prev' node point to the node before the reversed group
#         prev = dummy

#         # Modify the LL
#         while True:

#             # Define a node to traverse the list up to k and the counter
#             curr = prev.next
#             count = 0          

#             # Traverse the list to find the kth node
#             while curr and count < k:
#                 curr = curr.next
#                 count += 1

#             # Break Case: If we have fewer than k nodes (if count < k)
#             if count < k: 
#                 break

#             # Start reversing the k nodes
#             curr = prev.next # Starting point for the reverse
#             next_node = curr.next # Next node to reverse

#             # Reverse k nodes
#             for _ in range(1,k):

#                 temp = next_node.next
#                 next_node.next = prev.next
#                 prev.next = next_node
#                 curr.next = temp

#                 next_node = temp

#             # Move 'prev' to the end of the reversed group  
#             prev = curr

#         # Return the result holder
#         return dummy.next

#     # Testing the main func
#     node = reverseKGroup(head=head, k=k)


#     # # Testing the aux func
#     # node = reverse_ll(head=head)

#     while node:
#         print(node.val, end=', ')
#         node = node.next

#     '''
#     Notes: 
#         - Originally my intention was to solve the challenge with the reversal method in reverse_ll that
#             reverses the list all at one and leaves unconnected the reversed part so far to the rest of the
#             list in each iteration and only makes sense at the end of the reversal, but while revising the challenge
#             there was another way to reorder the list dynamically and that method will be explained below.

#             that way of solving the challenge was more clear to be understood than the initially planned.

#         - Either way, my intention to actually solve this would be functional with the solution at the bottom.
#     '''

#     # Dynamically reordering function
#     def reverse_list_reorder(head: Optional[ListNode]) -> Optional[ListNode]:

#         # Edge case: If the list is empty or has only one node, return as is
#         if not head or not head.next:
#             return head

#         # Dummy node to simplify pointer adjustments
#         dummy = ListNode(0)
#         dummy.next = head

#         # Start pointers
#         prev = dummy
#         curr = head
#         next_node = curr.next

#         # Loop to reorder and reverse the list
#         while next_node:
#             temp = next_node.next       # Keep track of the next node after next_node
#             next_node.next = prev.next  # Move next_node to the front
#             prev.next = next_node       # Point prev to the newly moved node
#             curr.next = temp            # Connect curr to the rest of the list
#             next_node = temp            # Move to the next node in the original order

#         return dummy.next
    

#     # Initial plan fixed and tested
#     def reverse_k_group(head: Optional[ListNode], k: int) -> Optional[ListNode]:

#         # Check if there are at least k nodes left to reverse
#         def has_k_nodes(curr: ListNode, k: int) -> bool:
#             count = 0
#             while curr and count < k:
#                 curr = curr.next
#                 count += 1
#             return count == k

#         # Define the pointers
#         dummy = ListNode(0)
#         dummy.next = head
#         prev = dummy

#         # Alter the list
#         while has_k_nodes(prev.next, k):

#             # Capure the start and the a node to capture the end of the group
#             start = prev.next
#             end = prev

#             # get to the end of the group
#             for _ in range(k):
#                 end = end.next

#             next_group = end.next   # Capture the rest of the list after the group
#             end.next = None         # Temporarily detach the k-group

#             # Reverse the current k-group using reverse_ll
#             reversed_group = reverse_ll(start)

#             # Reconnect the reversed group
#             prev.next = reversed_group
#             start.next = next_group

#             # Move prev to the end of the reversed group
#             prev = start

#         return dummy.next
        
#     'Done'

'''114. Flatten Binary Tree to Linked List'''
# def x():
    
#     from typing import Optional

#     # Definition for a binary tree node
#     class TreeNode:
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     # Case 1
#     tree = [1,2,5,3,4,None,6]
#     root = TreeNode(val=1, 
#                     left=TreeNode(val=2,
#                                   left=TreeNode(val=3),
#                                   right=TreeNode(val=4)),
#                     right=TreeNode(val=5,
#                                    right=TreeNode(val=6))
#                     )
#     # Output: [1,null,2,null,3,null,4,null,5,null,6]

#     '''
#     My Approach

#         Intuition:
            
#             - Handle Corner Case: No Node passed
           
#             - Create a 'dummy' head pointer into which the linked list will be built
#                 and a 'curr' pointer that will be located to in the 'right' pointer of the dummy.
            
#             - Define a preorder traversal function: 
#                 *This function will add each node to the curr's 'right' pointer.
#                 *And will also move the 'curr' pointer to the right to be located at the just added node.

#             - Reassign 'root' to the dummy's 'right' pointer
#     '''

#     'O(n) Approach'
#     def flatten(root: Optional[TreeNode]) -> None:

#         # Handle Corner case: ...
#         if not root:
#             return None
                
#         ll_layout = []

#         # Preorder traversal function definition
#         def preorder(node:TreeNode) -> None:

#             if not node:
#                 return    

#             ll_layout.append(node)
#             preorder(node=node.left)
#             preorder(node=node.right)
        

#         preorder(node=root)


#         for i in range(len(ll_layout)-1):

#             curr = ll_layout[i]
#             curr.left = None
#             curr.right = ll_layout[i+1]
               
    
#     # Testing
#     print(flatten(root=root))
        


#     'Optimized O(1) Space Solution'
#     def flatten(root: Optional[TreeNode]) -> None:

#         # Handle Corner case: No node passed
#         if not root:
#             return None
                
        
#         # Create a mutable container for curr so that changes are shared
#         curr = [None]  # Using a list to hold the current pointer
        

#         # Preorder traversal function definition
#         def preorder(node:TreeNode) -> None:

#             if not node:
#                 return
           
#             # Flatten the current node
#             if curr[0]:  # If curr[0] exists, link it to the current node
#                 curr[0].right = node
#                 curr[0].left = None

#             # Move curr to the current node
#             curr[0] = node
            
#             # Save the right subtree before recursion (because we modify the right pointer)
#             right_subtree = node.right
            
#             # Traverse left subtree first (preorder)
#             preorder(node.left)
            
#             # Traverse right subtree last
#             preorder(right_subtree)
        
#         # Traverse the root with the preorder function
#         preorder(node=root)


#     # Testing
#     print(flatten(root=root))
    

#     '''
#     Note: 
       
#        - Within a recursion inmutable object can not be affected out of the function, since Python creates a copy of them to work locally, the workaround here is to work with mutables (a list).

#        - The right sub-tree must be saved because is the next recursive call the right pointer will be modified and when it rolls back, the remaining right part will be lost otherwise.
    
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









