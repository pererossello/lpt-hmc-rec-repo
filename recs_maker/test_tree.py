import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from recs.grove import Grove

GROVE_DIR = './the_samples/EXP_B'
MyGrove = Grove(GROVE_DIR)

MyGrove.clear_grove()


trunks = MyGrove.add_n_trunks(2)

print(trunks[0].path)

# MyGrove.print_grove()
# MyGrove.save_grove_to_txt()
# MyGrove.add_branch([1])
# Tree.print_tree()

# print(Tree.tree.children[0].path)
#Tree.create_root()