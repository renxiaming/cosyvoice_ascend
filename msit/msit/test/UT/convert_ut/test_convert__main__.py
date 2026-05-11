import unittest
from unittest.mock import patch, MagicMock, call
import argparse

# msit/components/convert/model_convert/test___main__.py


from model_convert.__main__ import (
    parse_input_param,
    ModelConvertCommand,
    AieCommand,
    get_cmd_instance,
    MAX_READ_FILE_SIZE_32G,
)

class TestParseInputParam(unittest.TestCase):
    @patch("model_convert.__main__.ConvertConfig")
    def test_parse_input_param_success(self, mock_config):
        mock_config.return_value = "cfg"
        result = parse_input_param("m", "o", "soc")
        mock_config.assert_called_once_with(model="m", output="o", soc_version="soc")
        self.assertEqual(result, "cfg")

class TestModelConvertCommand(unittest.TestCase):
    @patch("model_convert.__main__.add_arguments")
    def test_add_arguments(self, mock_add_args):
        cmd = ModelConvertCommand(backend="atc", name="atc", help_info="help")
        parser = MagicMock()
        cmd.add_arguments(parser)
        mock_add_args.assert_called_once_with(parser, backend="atc")
        self.assertEqual(cmd.conf_args, mock_add_args.return_value)

    @patch("model_convert.__main__.execute_cmd")
    @patch("model_convert.__main__.gen_convert_cmd")
    def test_handle(self, mock_gen_cmd, mock_exec_cmd):
        cmd = ModelConvertCommand(backend="aoe", name="aoe", help_info="help")
        cmd.conf_args = "conf"
        args = MagicMock()
        mock_gen_cmd.return_value = "cmd"
        cmd.handle(args)
        mock_gen_cmd.assert_called_once_with("conf", args, backend="aoe")
        mock_exec_cmd.assert_called_once_with("cmd")

class TestAieCommand(unittest.TestCase):
    def setUp(self):
        self.cmd = AieCommand("aie", help_info="help")

    def test_add_arguments(self):
        parser = MagicMock()
        self.cmd.add_arguments(parser)
        calls = [
            call.add_argument("-gm", "--golden-model", dest="model", required=True, default=None, help="the path of the onnx model"),
            call.add_argument("-of", "--output-file", dest="output", required=True, default=None, help="Output file path&name(needn\'t .om suffix for ATC, need .om suffix for AIE)"),
            call.add_argument("-soc", "--soc-version", dest="soc_version", required=True, default=None, help="The soc version."),
        ]
        parser.assert_has_calls(calls, any_order=False)

    @patch("model_convert.__main__.logger")
    @patch("model_convert.__main__.Convert")
    @patch("model_convert.__main__.parse_input_param")
    @patch("model_convert.__main__.get_valid_write_path")
    @patch("model_convert.__main__.get_valid_read_path")
    def test_handle_success(self, mock_read, mock_write, mock_parse, mock_convert, mock_logger):
        args = argparse.Namespace(model="m", output="o", soc_version="soc")
        mock_read.return_value = "m_path"
        mock_write.return_value = "o_path"
        mock_parse.return_value = "cfg"
        mock_converter = MagicMock()
        mock_convert.return_value = mock_converter
        self.cmd.handle(args)
        mock_logger.info.assert_any_call("AIE start converting now")
        mock_read.assert_called_once_with("m", size_max=MAX_READ_FILE_SIZE_32G)
        mock_write.assert_called_once_with("o")
        mock_parse.assert_called_once_with("m_path", "o_path", "soc")
        mock_convert.assert_called_once_with("cfg")
        mock_converter.convert_model.assert_called_once()
        mock_logger.info.assert_any_call("AIE convert success")

    @patch("model_convert.__main__.logger")
    @patch("model_convert.__main__.parse_input_param", side_effect=ValueError("fail"))
    @patch("model_convert.__main__.get_valid_write_path")
    @patch("model_convert.__main__.get_valid_read_path")
    def test_handle_parse_input_param_fail(self, mock_read, mock_write, mock_parse, mock_logger):
        args = argparse.Namespace(model="m", output="o", soc_version="soc")
        mock_read.return_value = "m_path"
        mock_write.return_value = "o_path"
        self.cmd.handle(args)
        mock_logger.error.assert_called_with("fail")

    @patch("model_convert.__main__.logger")
    @patch("model_convert.__main__.Convert", return_value=None)
    @patch("model_convert.__main__.parse_input_param")
    @patch("model_convert.__main__.get_valid_write_path")
    @patch("model_convert.__main__.get_valid_read_path")
    def test_handle_convert_none(self, mock_read, mock_write, mock_parse, mock_convert, mock_logger):
        args = argparse.Namespace(model="m", output="o", soc_version="soc")
        mock_read.return_value = "m_path"
        mock_write.return_value = "o_path"
        mock_parse.return_value = "cfg"
        self.cmd.handle(args)
        mock_logger.error.assert_called_with("The object of 'convert' create failed.")

class TestGetCmdInstance(unittest.TestCase):
    def test_get_cmd_instance(self):
        cmd = get_cmd_instance()
        self.assertEqual(cmd.name, "convert")
        self.assertIn("convert tool", cmd.help_info)
        names = [c.name for c in cmd.children]
        self.assertIn("aie", names)
        self.assertIn("atc", names)
        self.assertIn("aoe", names)

if __name__ == "__main__":
    unittest.main()