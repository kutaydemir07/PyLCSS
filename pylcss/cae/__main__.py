"""Launch PyLCSS CAE directly."""
import sys
from pylcss.main import main
sys.argv[1:1] = ["--workspace", "cae"]
main()
