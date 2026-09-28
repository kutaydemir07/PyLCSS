"""Launch PyLCSS MDAO directly."""
import sys
from pylcss.main import main
sys.argv[1:1] = ["--workspace", "mdao"]
main()
