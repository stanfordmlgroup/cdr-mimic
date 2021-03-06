import util

from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False

        self.parser.add_argument('--phase', type=str, default='test', choices=('train', 'valid', 'test'),
                                 help='Phase to test on.')
        self.parser.add_argument('--results_dir', type=str, default='results/', help='Save dir for test results.')
