'''
CHALLENGES INDEX

297. Serialize and Deserialize Binary Tree (DFS) (BFS)
300. Longest Increasing Subsequence (DP)
315. Count of Smaller Numbers After Self - Partially solved





*DS: Dynamic Programming
*RC: Recursion
*TP: Two-pointers
*FCD: Floyd's cycle detection
*PS: Preffix-sum
*SW: Sliding-Window
*FCD: Floyd's Cycle Detection (Hare & Tortoise)


(3)
'''




'''297. Serialize and Deserialize Binary Tree'''

# # Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


# Input

# # Case 1
# root_map = [1,2,3,None,None,4,5]

# root = TreeNode(1)
# two, three = TreeNode(2), TreeNode(3)
# four, five = TreeNode(4), TreeNode(5)

# root.left, root.right = two, three
# three.left, three.right = four, five

# Custom Case
# root_map = [4,-7,-3,None,None,-9,-3,9,-7,-4,None,6,None,-6,-6,None,None,0,6,5,None,9,None,None,-1,-4,None,None,None,-2]

# root = TreeNode(4)
# two, three = TreeNode(-7), TreeNode(-3)
# root.left, root.right = two, three

# four, five = TreeNode(-9), TreeNode(-3)
# three.left, three.right = four, five

# six, seven, eight = TreeNode(9), TreeNode(-7), TreeNode(-4)
# four.left, four.right = six, seven
# five.left = eight

# nine, ten = TreeNode(6), TreeNode(-6)
# seven.left = nine
# eight.right = ten

# eleven, twelve, thirteen = TreeNode(-6), TreeNode(0), TreeNode(6)
# nine.left, nine.right = eleven, twelve
# ten.left = thirteen

# fourteen, fifteen = TreeNode(5), TreeNode(-2)
# thirteen.left, thirteen.right = fourteen, fifteen

# sixteen, seventeen = TreeNode(9), TreeNode(-1)
# fourteen.left, fourteen.right = sixteen, seventeen

# eighteen = TreeNode(-4)
# seventeen.left = eighteen

# Output: [4,-7,-3,null,null,-9,-3,9,-7,-4,null,6,null,-6,-6,null,null,0,6,5,null,9,null,null,-1,-4,null,null,null,-2]




'Solution'

# class Codec:

#     def serialize(self, root):
#         """
#         Encodes a tree to a single string.
        
#         :type root: TreeNode
#         :rtype: str
#         """

#         # Handle corner case
#         if not root:
#             return ''

#         queue = [root]
#         visited = []

#         while queue:

#             node = queue.pop(0)

#             if node:
#                 visited.append(str(node.val))
#                 queue.extend([node.left, node.right])

#             else:
#                 visited.append('None')        

#         return ','.join(visited)
        

#     def deserialize(self, data):
#         """
#         Decodes your encoded data to tree.
        
#         :type data: str
#         :rtype: TreeNode
#         """

#         # Handle corner case
#         if not data:
#             return
        
#         # Transform data into a valid input for the tree
#         data = [int(x) if x != 'None' else None for x in data.split(',')]

#         # Initilize the root
#         root = TreeNode(data[0])

#         # Populate the tree
#         index = 1
#         queue = [root]

#         while index < len(data) and queue:

#             node = queue.pop(0)

#             if data[index]:

#                 node.left = TreeNode(data[index])
#                 queue.append(node.left)
            
#             index += 1
            
#             if data[index]:

#                 node.right = TreeNode(data[index])
#                 queue.append(node.right)
            
#             index += 1

#         return root


# # Testing

# ser = Codec()
# deser = Codec()
# ans = deser.serialize(root=root)

# # print(ans)


# # Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

# # print(ans)


# # Auxiliary pretty print function
# def pretty_print_bst(node, prefix="", is_left=True):

#     if not node:
#         node = root
    
#     if not node:
#         print('Empty Tree')
#         return


#     if node.right is not None:
#         pretty_print_bst(node.right, prefix + ("│   " if is_left else "    "), False)

#     print(prefix + ("└── " if is_left else "┌── ") + str(node.val))

#     if node.left is not None:
#         pretty_print_bst(node.left, prefix + ("    " if is_left else "│   "), True)


# pretty_print_bst(node=root)
# print('\n\n\n\n')
# pretty_print_bst(node=ans)

'Done'




'''300. Longest Increasing Subsequence'''

# # Input

# # Case 1
# nums = [10,9,2,5,3,7,101,18]
# # Output: 4

# # Case 2
# nums = [0,1,0,3,2,3]
# # Output: 4

# # Case 3
# nums = nums = [7,7,7,7,7,7,7]
# # Output: 1


'DP Solution'

# def lengthOfLIS(nums: list[int]) -> int:    
    
#     # Handle corner case
#     if not nums:
#         return 0
    

#     # Initialize the dp array
#     dp = [1] * len(nums)

#     # Iterate through the elements of the list, starting from the second
#     for i in range(1, len(nums)):

#         for j in range(i):

#             if nums[i] > nums[j]:
#                 dp[i] = max(dp[i], dp[j]+1)


#     return max(dp)

'Done'




'''315. Count of Smaller Numbers After Self'''

# Input

# # Case 1
# nums = [5,2,6,1]
# # Output: [2,1,1,0]

# # Case 1
# nums = [-1,-1]
# # Output: [0,0]


'My Approach (Brute forcing)'

# def count_smaller(nums: list[int]) -> list[int]:

#     # Handle corner case
#     if len(nums) == 1:
#         return [0]
    

#     # Set the min value of the group
#     min_num = min(nums)
    

#     # Initialize the result holder
#     result = []

#     for x,num in enumerate(nums):

#         # corner case: if the number is the smallest of the group or the right most one, no smaller numbers after it
#         if num == min_num or num == nums[-1]:
#             result.append(0)
        
#         else:

#             # Define a sublist with all elements to the right of the current one
#             sublist = nums[x+1:]

#             # Count how many of those are smaller than the current one
#             count = len([x for x in sublist if x<num])

#             # Add that result to the holder
#             result.append(count)
            
#     return result
     

# print(count_smaller(nums=nums))

'Note: This approach met up to 79% o the cases'





'''xxx'''









