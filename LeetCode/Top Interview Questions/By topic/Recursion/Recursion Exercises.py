'''
CHALLENGES INDEX

2. Add Two Numbers (LL) (RC)
21. Merge Two Sorted Lists (LL) (RC)
50. Pow(x, n) (RC)
206. Reverse Linked List (LL) (RC)
234. Palindrome Linked List (LL) (RC) (TP)


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


'2. Add Two Numbers'
# def x():

#     # Definition for singly-linked list.
#     class ListNode(object):
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input
#     l1 = ListNode(2, ListNode(4, ListNode(3)))
#     l2 = ListNode(5, ListNode(6, ListNode(4)))


#     # LeetCode Editorial solution
#     def addTwoNumbers(l1, l2):

#         dummyHead = ListNode(0)

#         curr = dummyHead

#         carry = 0

#         while l1 != None or l2 != None or carry != 0:

#             l1Val = l1.val if l1 else 0
#             l2Val = l2.val if l2 else 0

#             columnSum = l1Val + l2Val + carry

#             carry = columnSum // 10

#             newNode = ListNode(columnSum % 10)

#             curr.next = newNode
#             curr = newNode

#             l1 = l1.next if l1 else None
#             l2 = l2.next if l2 else None

#         return dummyHead.next

#     result = addTwoNumbers(l1, l2)



#     # My version of the solution

#     # 1st list processing
#     l1_list = []
#     l1_next_node = l1

#     while l1_next_node is not None:

#         l1_list.append(l1_next_node.val)
#         l1_next_node = l1_next_node.next

#     l1_num = str()

#     for num in l1_list:
#         l1_num += str(num)

#     l1_num = int(l1_num[-1::-1])


#     # 2nd list processing
#     l2_list = []
#     l2_next_node = l2

#     while l2_next_node is not None:

#         l2_list.append(l2_next_node.val)
#         l2_next_node = l2_next_node.next

#     l2_num = str()

#     for num in l2_list:
#         l2_num += str(num)

#     l2_num = int(l2_num[-1::-1])


#     # Result outputting

#     lr_num = l1_num + l2_num
#     lr_str = str(lr_num)[-1::-1]

#     lr_llist = ListNode()
#     curr = lr_llist

#     for i in lr_str:

#         new_node = ListNode(i)

#         curr.next = new_node
#         curr = new_node


#     # Validating
#     while lr_llist is not None:
#         print(lr_llist.val)
#         lr_llist = lr_llist.next

'21. Merge Two Sorted Lists'
# def x():

#     # Base
#     class ListNode(object):
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input

#     # 1st Input
#     #List 1
#     one1, two1, three1 = ListNode(1), ListNode(2), ListNode(4)
#     one1.next, two1.next = two1, three1

#     #List 2
#     one2, two2, three2 = ListNode(1), ListNode(3), ListNode(4)
#     one2.next, two2.next = two2, three2


#     # 2nd Input
#     #List 1
#     one1, two1, three1 = ListNode(4), ListNode(3), ListNode(4)
#     one1.next, two1.next = two1, three1

#     #List 2
#     one2, two2, three2 = ListNode(1), ListNode(0), ListNode(50)
#     one2.next, two2.next = two2, three2

#     # My Approach
#     def mergeTwoLists(list1:ListNode, list2:ListNode) -> ListNode:

#         if list1.val == None and list2.val != None:
#             return list2
        
#         if list2.val == None and list1.val != None:
#             return list1
        
#         if list1.val == None and list2.val == None:
#             return ListNode(None)


#         head = ListNode()
#         curr_res = head

#         curr1, curr2 = list1, list2

#         while True:

#             if curr1 != None and curr2 != None:
                
#                 if curr1.val <= curr2.val:
#                     curr_res.next = curr1
#                     curr_res = curr_res.next
#                     curr1 = curr1.next     
                    
#                 else:
#                     curr_res.next = curr2
#                     curr_res = curr_res.next
#                     curr2 = curr2.next                   

#             elif curr1 != None:
#                 curr_res.next = curr1
#                 curr_res = curr_res.next
#                 curr1 = curr1.next

#             elif curr2 != None:
#                 curr_res.next = curr2
#                 curr_res = curr_res.next
#                 curr2 = curr2.next
            

#             if curr1 == None and curr2 == None:
#                 break

#         return head.next

#     # Testing
#     res = []
#     res_node = mergeTwoLists(one1, one2)

#     while res_node != None:

#         res.append(res_node.val)
#         res_node = res_node.next


#     print(res)

#     'Notes: it works!'

'50. Pow(x, n)'
# def x():

#     # Input

#     # Case 1
#     x = 2.00000
#     n = 10
#     # Output: 1024.00000

#     # Case 2
#     x = 2.10000
#     n = 3
#     # Output: 9.26100

#     # Case 3
#     x = 2.00000
#     n = -2
#     # Output: 0.25000

#     # Custom Case
#     x = 0.00001
#     n = 2147483647
#     # Output: ...

#     # My Approach
#     def myPow(x, n):

#         if x == 0:
#             return 0
        
#         if n == 0:
#             return 1

#         res = 1

#         for _ in range(abs(n)):
#             res *= x

#         if n > 0:
#             return f'{res:.5f}'
        
#         else:        
#             return f'{(1/res):.5f}'


#     print(myPow(x, n))

#     'Notes: it works, but broke memory with the case: x = 0.00001, n=2147483647, it is 95% of the solution'

#     # Another Approach
#     def myPow(x: float, n: int) -> float:

#         b = n

#         if x == 0:
#             return 0
        
#         elif b == 0:
#             return 1
        
#         elif b < 0:
#             b = -b
#             x = 1 / x
        

#         a = 1

#         while b > 0:

#             if b % 2 == 0:
#                 x = x * x
#                 b = b // 2

#             else:
#                 b = b - 1
#                 a = a * x
                
#         return a

#     print(myPow(x, n))

#     '''
#     Notes: 
#         This solution takes advantage of the property x^(2n) = (x^2)^n, 
#         saving a lot of time reducing in half the calculations each time the exponent is even.
#     '''

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


















