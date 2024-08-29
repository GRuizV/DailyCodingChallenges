'''
CHALLENGES INDEX

2. Add Two Numbers


(xx)

'''




'2. Add Two Numbers'

# # Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

# # Input
# l1 = ListNode(2, ListNode(4, ListNode(3)))
# l2 = ListNode(5, ListNode(6, ListNode(4)))


# # LeetCode Editorial solution
# def addTwoNumbers(l1, l2):

#     dummyHead = ListNode(0)

#     curr = dummyHead

#     carry = 0

#     while l1 != None or l2 != None or carry != 0:

#         l1Val = l1.val if l1 else 0
#         l2Val = l2.val if l2 else 0

#         columnSum = l1Val + l2Val + carry

#         carry = columnSum // 10

#         newNode = ListNode(columnSum % 10)

#         curr.next = newNode
#         curr = newNode

#         l1 = l1.next if l1 else None
#         l2 = l2.next if l2 else None

#     return dummyHead.next

# result = addTwoNumbers(l1, l2)



# # My version of the solution

# # 1st list processing
# l1_list = []
# l1_next_node = l1

# while l1_next_node is not None:

#     l1_list.append(l1_next_node.val)
#     l1_next_node = l1_next_node.next

# l1_num = str()

# for num in l1_list:
#     l1_num += str(num)

# l1_num = int(l1_num[-1::-1])


# # 2nd list processing
# l2_list = []
# l2_next_node = l2

# while l2_next_node is not None:

#     l2_list.append(l2_next_node.val)
#     l2_next_node = l2_next_node.next

# l2_num = str()

# for num in l2_list:
#     l2_num += str(num)

# l2_num = int(l2_num[-1::-1])


# # Result outputting

# lr_num = l1_num + l2_num
# lr_str = str(lr_num)[-1::-1]

# lr_llist = ListNode()
# curr = lr_llist

# for i in lr_str:

#     new_node = ListNode(i)

#     curr.next = new_node
#     curr = new_node


# # Validating
# while lr_llist is not None:
#     print(lr_llist.val)
#     lr_llist = lr_llist.next




'xxx'