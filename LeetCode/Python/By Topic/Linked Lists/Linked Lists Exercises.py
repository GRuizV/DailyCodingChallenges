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
92. Reverse Linked List II (LL)
82. Remove Duplicates from Sorted List II (LL)
61. Rotate List (LL)
86. Partition List (LL) (TP)


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


(20)
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
    
#     from typing import Optional

#     # Input
#     # Case 1
#     head = [1,2,3,4,5]
#     n = 2
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     head.next.next.next = ListNode(4)
#     head.next.next.next.next = ListNode(5)
#     # Output: [1,2,3,5]
    
#     # Case 2
#     head = [1]
#     n = 1
#     head = ListNode(1)
#     # Output: []

#     # Case 3
#     head = [1,2]
#     n = 1
#     head = ListNode(1)
#     head.next = ListNode(2)
#     # Output: [1]

#     # Case 4
#     head = [1,2]
#     n = 2
#     head = ListNode(1)
#     head.next = ListNode(2)
#     # Output: [2]

#     # Case 5
#     head = [1,2,3]
#     n = 3
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     # Output: [2,3]


#     '''
#     My Approach (Double passing)

#         Intuition:
            
#             - Handle corner cases: if no node passed return the input.
#             - Create a dummy node initialized at 0.
#             - Point the 'next' poiner of dummy to the 'head'.
#             - Initialize a counter (for the nodes) at 1.
#             - Create a curr node holder initialized at dummy's 'next' pointer.
#             - In a While Loop (while curr):
#                 + curr = curr.next.
#                 + counter adds 1.
#             - Substract 'n' to 'counter'.
#             - Initialize an index 'i' at 1.
#             - Reset 'curr' back to dummy's next pointer.
#             - Initialize a 'prev' node holder at 'dummy'.
#             - In a While Loop (while curr):
#                 + if 'i' is the current 'counter' number:
#                     * prev 'next' pointer assigned to curr 'next' pointer if exists,
#                         otherwise, prev 'next' should be assigned to None.
#                     * Break
                
#                 + curr = curr.next.
#                 + counter adds 1.
            
#             - Return dummy's 'next'.

#     '''

#     def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:

#         # Handle Corner case
#         if not head:
#             return head
        
#         # Create a dummy node initialized at 0
#         dummy = ListNode(0)

#         # Point the 'next' poiner of dummy to the 'head'
#         dummy.next = head

#         # Initialize a counter at 1.
#         counter = 1

#         # Create a curr node holder initialized at dummy's 'next'
#         curr = dummy.next

#         # Probe the list
#         while curr:
#             curr = curr.next
#             counter += 1
        
#         # Substract 'n' to 'counter'
#         counter -= n

#         # Initialize an index 'i' at 1
#         i = 1

#         # Reset 'curr' back to dummy's next pointer
#         curr = dummy.next

#         # Initialize a 'prev' node holder at 'dummy'
#         prev = dummy

#         # Traverser the list again
#         while curr:

#             if i == counter:

#                 prev.next = curr.next if curr.next else None
            
#                 break
            
#             prev = curr
#             curr = curr.next
#             i += 1

#         # Return the node next to dummy
#         return dummy.next

#     # Testing
#     dummy = removeNthFromEnd(head=head, n=n)
#     res = []

#     while dummy:
#         res.append(dummy.val)
#         dummy = dummy.next
    
#     print(res)

#     '''
#     Note: 
    
#         It worked just fine!
#             Runtime: 0ms (100%)
#             Memory: 17.9mb (11.2%)

#         But... as follow up, the problem asks if we could do it in just one pass,
#         and I think we do, with the help of a Hash-Table.

#     '''




#     '''
#     My Approach (Hash Table)

#         Intuition:
            
#             - Handle corner cases: if no node passed return the input.
#             - Create a dummy node initialized at 0.
#             - Point the 'next' poiner of dummy to the 'head'.
#             - Initialize a counter (for the nodes) at 0.
#             - Initialize an empty dictionary holder.
#             - Create a curr node holder initialized at dummy's.
#             - In a While Loop (while curr):
#                 + Add an entry to the dictionary with counter as key and the 'curr' node as value.
#                 + curr = curr.next.
#                 + counter adds 1.
#             - Substract 'n' to 'counter'.
#             - Assign 'counter + 1' entry of the dict to the 'counter - 1' entry of the dict, if
#                 'counter + 1' is less than the length of the dictionary, otherwise assign 'None'
#                 to 'counter - 1' entry of the dict.
#             - Return dummy's 'next'.

#     '''

#     def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:

#         # Handle Corner case
#         if not head:
#             return head
        
#         # Create a dummy node initialized at 0
#         dummy = ListNode(0)

#         # Point the 'next' poiner of dummy to the 'head'
#         dummy.next = head

#         # Initialize a counter at 1.
#         counter = 0

#         # Create a curr node holder initialized at dummy's 'next'
#         curr = dummy

#         # Initialize an empty dictionary holder
#         dic = {}

#         # Probe the list
#         while curr:
#             dic[counter] = curr
#             curr = curr.next
#             counter += 1
        
#         # Substract 'n' to 'counter'
#         counter -= n

#         # # Make the modification to the list through the dictionary
#         dic[counter-1].next = dic[counter+1] if counter+1 < len(dic) else None
        
#         # Return the node next to dummy
#         return dummy.next

#     # Testing
#     dummy = removeNthFromEnd(head=head, n=n)
#     res = []

#     while dummy:
#         res.append(dummy.val)
#         dummy = dummy.next
    
#     print(res)

#     '''
#     Note: 
    
#         It worked just fine!
#             Runtime: 0ms (100%)
#             Memory: 17.96mb (11.2%)

#         It is made in a single passing but at the cost of using O(n) of memory, so that's not the optimal wat to do it.

#         Actually with Floyd's Detection Cycle, and with a Two-Pointer's approach this could be solved in just one passing (Time: O(n)) and with constant space (Space: O(1)).
#         This would be actually the same of the first solution but more elegant.

#     '''




#     '''
#     Two-Pointer's approach
    
#         Explanation:
#             - Maintain two pointers: fast and slow.
#             - Move fast pointer n steps ahead.
#             - Move both fast and slow together until fast reaches the end.
#             - Slow now points to the node before the one to be deleted.
#             - Modify slow.next to remove the nth node.

#     '''

#     def removeNthFromEnd(head: Optional[ListNode], n: int) -> Optional[ListNode]:

#         # Create a dummy node before head to simplify edge cases
#         dummy = ListNode(0)
#         dummy.next = head
#         fast = slow = dummy

#         # Move fast pointer `n + 1` steps ahead (so slow stops at the node before the target)
#         for _ in range(n + 1):
#             fast = fast.next

#         # Move both fast and slow until fast reaches the end
#         while fast:
#             fast = fast.next
#             slow = slow.next

#         # Remove the nth node
#         slow.next = slow.next.next

#         return dummy.next

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

'''92. Reverse Linked List II'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     head = [1,2,3,4,5]
#     left = 2
#     right = 4
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     head.next.next.next = ListNode(4)
#     head.next.next.next.next = ListNode(5)
#     # Output: [1,4,3,2,5]

#     # Case 2
#     head = [5]
#     left = 1
#     right = 1
#     head = ListNode(5)
#     # Output: [5]

#     # Case 3
#     head = [3,5]
#     left = 1
#     right = 2
#     head = ListNode(3)
#     head.next = ListNode(5)
#     # Output: [5,3]

#     # Case 4
#     head = [1,2,3,4,5]
#     left = 1
#     right = 5
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     head.next.next.next = ListNode(4)
#     head.next.next.next.next = ListNode(5)
#     # Output: [5,4,3,2,1]

    
#     '''
#     Explanation

#         * Step 1: Handle Corner Cases
#         * Step 2: Create a Dummy node before head.
#         * Step 3: Move the 'prev' holder just before the left input.
#         * Step 4: Reverse the portion between left and right.

#         - Return dummy's next.

       
#     ''' 

#     def reverseBetween(head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:

#         # Step 1: Handle edge cases
#         if not head or left == right:
#             return head

#         # Step 2: Create a dummy node before head
#         dummy = ListNode(0)
#         dummy.next = head
#         prev = dummy

#         # Step 3: Move `prev` to the node before `left`
#         for _ in range(left - 1):
#             prev = prev.next

#         # Step 4: Reverse the sublist
#         curr = prev.next  # The `left` node
#         next_node = None

#         for _ in range(right - left):
#             next_node = curr.next
#             curr.next = next_node.next
#             next_node.next = prev.next
#             prev.next = next_node

#         return dummy.next

#     # Testing
#     curr = reverseBetween(head=head, left=left, right=right)
#     res = []
#     while curr:
#         res.append(curr.val)
#         curr = curr.next
#     print(res)

#     '''Note: Done'''

'''82. Remove Duplicates from Sorted List II'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     head = [1,2,3,3,4,4,5]
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     head.next.next.next = ListNode(3)
#     head.next.next.next.next = ListNode(4)
#     head.next.next.next.next.next = ListNode(4)
#     head.next.next.next.next.next.next = ListNode(5)
#     # Output: [1,2,5]

#     # Case 2
#     head = [1,1,1,2,3]
#     head = ListNode(1)
#     head.next = ListNode(1)
#     head.next.next = ListNode(1)
#     head.next.next.next = ListNode(2)
#     head.next.next.next.next = ListNode(3)
#     # Output: [2,3]

#     # Case 3
#     head = [1,1,1,2,3,3,3,4,4,5]
#     head = ListNode(1)
#     head.next = ListNode(1)
#     head.next.next = ListNode(1)
#     head.next.next.next = ListNode(2)
#     head.next.next.next.next = ListNode(3)
#     head.next.next.next.next.next = ListNode(3)
#     head.next.next.next.next.next.next = ListNode(3)
#     head.next.next.next.next.next.next.next = ListNode(4)
#     head.next.next.next.next.next.next.next.next = ListNode(4)
#     head.next.next.next.next.next.next.next.next.next = ListNode(5)
#     # Output: [2,5]

#     # Case 4
#     head = [1,2,3,4,5]
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     head.next.next.next = ListNode(4)
#     head.next.next.next.next = ListNode(5)
#     # Output: [1,2,3,4,5]

#     # Case 5
#     head = []
#     head = None

#     # Case 6
#     head = [1,1,1,1,1]
#     head = ListNode(1)
#     head.next = ListNode(1)
#     head.next.next = ListNode(1)
#     head.next.next.next = ListNode(1)
#     head.next.next.next.next = ListNode(1)
#     # Output: []


#     '''
#     My Approach (Linked Lists)

#         Intuition:
            
#             - Handle corner case: No node is passed.
#             - Initialize a 'dummy' node holder at 0.
#             - Point dummy's next pointer to the head of the list.
#             - Initialize a 'prev' node holder at dummy.
#             - Initialize a 'curr' node holder pointing at the head of list.
#             - In a While Loop (while curr):
                
#                 + If the node next to the current exists, and that node's value is the same as the curr:

#                     * While the node next to the current exists, and that node's value is the same as the curr:
#                         - Move 'curr' one node ahead.

#                     * Make prev point to curr's next.
                
#                 + else: 
                    
#                     * Move 'prev' one node ahead.

#                 + Move curr one node ahead               
                
#             - Return dummy's next.
#     '''

#     def deleteDuplicates(head: Optional[ListNode]) -> Optional[ListNode]:

#         # Handle Corner case: No node passed
#         if not head:
#             return None
        
#         # Initialize a 'dummy' node holder
#         dummy = ListNode(0)

#         # Point dummy's next pointer to the head of the list
#         dummy.next = head

#         # Initialize a 'prev' node holder
#         prev = dummy

#         #  Initialize a 'curr' node holder 
#         curr = head
        
#         # Process the list
#         while curr:
            
#             # If the node next to the current exists, and that node's value is the same as the curr
#             if curr.next and curr.val == curr.next.val:
                
#                 # While the same conditional
#                 while curr.next and curr.val == curr.next.val:
#                     curr = curr.next    # Move curr one node ahead

#                 # Connect prev with the node next to curr
#                 prev.next = curr.next
           
#             # Otherwise
#             else:

#                 # Move prev one node ahead          
#                 prev = prev.next

#             # Move curr one node ahead
#             curr = curr.next

#         # Return the node next to dummy
#         return dummy.next

#     # Testing
#     dummy = deleteDuplicates(head=head)
#     res = []

#     while dummy:
#         res.append(dummy.val)
#         dummy = dummy.next
    
#     print(res)

#     '''Note: Done'''

'''61. Rotate List'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     head = [1,2,3,4,5],
#     k = 2
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     head.next.next.next = ListNode(4)
#     head.next.next.next.next = ListNode(5)
#     # Output: [4,5,1,2,3]

#     # Case 2
#     head = [0,1,2]
#     k = 4
#     head = ListNode(0)
#     head.next = ListNode(1)
#     head.next.next = ListNode(2)
#     # Output: [2,0,1]

#     # Case 3
#     head = []
#     k = 0
#     head = None
#     # Output: []

#     # Case 4
#     head = [1]
#     k = 0
#     head = ListNode(1)
#     # Output: [1]

#     # Case 5
#     head = [1,2]
#     k = 0
#     head = ListNode(1)
#     head.next = ListNode(2)
#     # Output: [1,2]

#     # Case 6
#     head = [1,2]
#     k = 1
#     head = ListNode(1)
#     head.next = ListNode(2)
#     # Output: [2,1]

#     # Case 7
#     head = [1,2]
#     k = 2
#     head = ListNode(1)
#     head.next = ListNode(2)
#     # Output: [1,2]
    

#     '''
#     My Approach 

#         Intuition:
            
#             Handle corner cases:
#                 + No node passed.
#                 + k = 0.
            
#             Create initial Variables                
#             - Initialize an int holder 'n' at 0 (Which will hold the length of the list).
#             - Initialize two ListNode holders at 0 named 'dummy' and 'sec_part'.
#             - Initialize a 'curr' List node holder at 'head' (This will be used to move around the list).

#             Probe the list
#             - In a While Loop (While 'curr' exists):
#                 + Move 'curr' one node ahead.
#                 + Add 1 to n.
            
#             Calculate the rotation start node: 
#             - Set 'k' to k modulo n.
#             - Initialize a variable 'ini_pos' at n - k 
            
#             Early Exit
#             if ini_pos is 0: return the head of the list as it is
            
#             List Modification
#             - Reset 'curr' back to 'head'.
#             - Point 'sec_part' next attribute to 'head'.
#             - In a for loop (for _ in range(ini_pos)):
#                 + Move 'curr' one node ahead.
#             - Point 'dummy' next attribute to 'curr.next'.
#             - Point 'curr.next' to None.
#             - Set 'curr' to 'dummy'.
#             - In a for loop (for _ in range(k)):
#                 + Move 'curr' one node ahead.
#             - Connect the end of the last piece of the partial list that starts at 'dummy' to the 'sec_part'
#                 as: curr.next = sec_part

            
#             - Return dummy's next.
#     '''

#     def rotateRight(head: Optional[ListNode], k: int) -> Optional[ListNode]:

#         # Handle Corner cases: Not node or k is 0
#         if not head or k==0:
#             return head

#         # Initialize a holder 'n'
#         n = 0
        
#         # Initialize two ListNode holders at 0
#         dummy = ListNode(0)
#         sec_part = ListNode(0)

#         # Initialize a 'curr' ListNode holder at 'head'
#         curr = head

#         # Probe the list
#         while curr:
#             curr = curr.next
#             n += 1

#         # Set 'k' to k modulo n 
#         k %= n

#         # Initialize a variable 'ini_pos' at n - k
#         ini_pos = n - k

#         # Early Exit Guard
#         if k == 0:
#             return head

#         # Reset 'curr' back to 'head'.
#         curr = head

#         # Point 'sec_part' next attribute to 'head'
#         sec_part.next = head

#         # Get to the rotation point in the list
#         for _ in range(ini_pos-1):
#             curr = curr.next

#         # Point 'dummy' next attribute to the new start of the rotated list
#         dummy.next = curr.next

#         # Close the tail of the second part of the rotated list
#         curr.next = None

#         # Reset back curr to the the point before the start of the new rotated list
#         curr = dummy

#         # Get to the end of the curr start of the rotated list
#         for _ in range(k):
#             curr = curr.next

#         # Connect the two parts together
#         curr.next = sec_part.next

#         # Return dummy's next
#         return dummy.next

#     # Testing
#     dummy = rotateRight(head=head, k=k)
#     res = []

#     while dummy:
#         res.append(dummy.val)
#         dummy = dummy.next
    
#     print(res)

#     '''Note: Done, but there is a simplified version of it'''




#     '''
#     Simplified Approach
#     '''

#     def rotateRight(head: Optional[ListNode], k: int) -> Optional[ListNode]:

#         # Handle Corner Cases: Empty list or no rotation needed
#         if not head or not head.next or k == 0:
#             return head

#         # Step 1: Compute the length of the list
#         n = 1
#         tail = head
#         while tail.next:
#             tail = tail.next
#             n += 1

#         # Step 2: Adjust `k` to avoid redundant rotations
#         k %= n
#         if k == 0:  
#             return head  # No rotation needed if `k` is a multiple of `n`

#         # Step 3: Find the new head (n-k from the start)
#         prev = head
#         for _ in range(n - k - 1):
#             prev = prev.next

#         # Step 4: Perform the rotation
#         new_head = prev.next
#         prev.next = None
#         tail.next = head  # Connect original tail to original head

#         return new_head
    
#     '''Notes: Now is done and pretty!'''

'''86. Partition List'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     head = [1,4,3,2,5,2]
#     x = 3
#     head = ListNode(1)
#     head.next = ListNode(4)
#     head.next.next = ListNode(3)
#     head.next.next.next = ListNode(2)
#     head.next.next.next.next = ListNode(5)
#     head.next.next.next.next.next = ListNode(2)
#     # Output: [1,2,2,4,3,5]

#     # Case 2
#     head = [2,1]
#     x = 2
#     head = ListNode(2)
#     head.next = ListNode(1)
#     # Output: [1,2]

#     # Case 3
#     head = [1,2,3]
#     x = 3
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     # Output: [1,2,3]

#     # Case 4
#     head = [1,2,3]
#     x = 1
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     # Output: [1,2,3]

#     # Case 5
#     head = [1,2,3]
#     x = 2
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     # Output: [1,2,3]

#     # Case 6
#     head = [1,2,3]
#     x = 4
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     # Output: [1,2,3]

#     # Case 7
#     head = [1,2,3]
#     x = 0
#     head = ListNode(1)
#     head.next = ListNode(2)
#     head.next.next = ListNode(3)
#     # Output: [1,2,3]

#     # Case 8
#     head = []
#     x = 0
#     head = None

    
#     '''
#     My Approach (Two Pointer)

#         Intuition:
            
#             - Handle corner case: No node.
#             - Initialize two listnode holders, 'fp' (first part) and 'sp' (second part) both at 0.
#             - Initialize three variables 'curr', 'fp_curr' and 'sp_curr' at 'head', 'fp' and 'sp' respectively.
            
#             - In a while loop (while there are node in the head list [while curr]):
#                 + if curr.val < x:
#                     * Assign fp_curr next pointer to curr
#                     * Move fp_curr one node ahead.
#                 + else:
#                     * Assign sp_curr next pointer to curr
#                     * Move sp_curr one node ahead. 
                
#                 + Move curr one node ahead.

#             - Set fp_curr next pointer to sp next's next pointer.
#             - Set sp_curr next pointer to None.

#             return fp next's pointer

#     '''

#     def partition(head: Optional[ListNode], x: int) -> Optional[ListNode]:

#         # Handle Corner case: No head
#         if not head:
#             return None

#         # Initialize listnode holders at 0
#         fp = ListNode(0)
#         sp = ListNode(0)

#         # Initialize mobile variables
#         curr, fp_curr, sp_curr = head, fp, sp

#         # Process the list
#         while curr:

#             # If the value is less than the partition pivot
#             if curr.val < x:
                
#                 # Assign the node to the first part (fp)
#                 fp_curr.next = curr

#                 # Move the first part mobile variable one node ahead
#                 fp_curr = fp_curr.next

#             # Otherwise
#             else:
                
#                 # Assign the node to the second part (sp)
#                 sp_curr.next = curr

#                 # Move the second part mobile variable one node ahead
#                 sp_curr = sp_curr.next

#             # Move curr one node ahead
#             curr = curr.next
        

#         # Join the two parts together
#         fp_curr.next = sp.next

#         # Close down the tail of the second part
#         sp_curr.next = None

#         # Return the node next to the fp node holder
#         return fp.next

#     # Testing
#     dummy = partition(head=head, x=x)
#     res = []

#     while dummy:
#         res.append(dummy.val)
#         dummy = dummy.next
    
#     print(res)

#     '''Note: This solution worked just fine. '''






