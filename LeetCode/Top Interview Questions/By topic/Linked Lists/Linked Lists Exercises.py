'''
CHALLENGES INDEX

2. Add Two Numbers (LL) (RC)
19. Remove Nth Node From End of List (LL) (TP)
21. Merge Two Sorted Lists (LL) (RC)
23. Merge k Sorted Lists (LL) (DQ) (Heap) (Sorting)


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













