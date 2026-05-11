import os
import unittest
from collections import defaultdict
import subprocess
from unittest.mock import patch, MagicMock
import pandas as pd

from msit_prof.analyze.autofuse.single_op_analyze import SingleOpAnalyzer, OpInfo


class TestOpInfo(unittest.TestCase):
    def test_init(self):
        op_name = "test_op"
        op_type = "test_type"
        op_info = OpInfo(op_name, op_type)
        self.assertEqual(op_info.op_name, op_name)
        self.assertEqual(op_info.op_type, op_type)

        op_name = "test_op"
        op_info = OpInfo(op_name)
        self.assertEqual(op_info.op_name, op_name)
        self.assertEqual(op_info.op_type, "unknown")


class TestSingleOpAnalyzer(unittest.TestCase):
    def setUp(self):
        self.args = MagicMock()
        self.args.fused = "fused.csv"
        self.args.origin = "origin.csv"
        self.args.output = "output_dir"
        self.args.ops_graph = "ge_graph_path"
        self.analyzer = SingleOpAnalyzer(self.args)
        self.analyzer.fuse_df = pd.DataFrame({
            'Op Name': ['fuse_op1', 'fuse_op2'],
            'OP Type': ['AscBackend', 'FusedAscBackend'],
            'Task Duration(us)': [100, 200]
        })
        self.analyzer.origin_df = pd.DataFrame({
            'Op Name': ['op1', 'op2', 'op3'],
            'OP Type': ['Conv2D', 'Reshape', 'FusedAscBackend'],
            'Input Data Types': ['FLOAT', 'FLOAT', 'FLOAT'],
            'Input Shapes': [512, 512, 512],
            'Output Data Types': ['FLOAT', 'FLOAT', 'FLOAT'],
            'Output Shapes': [512, 512, 512],
            'Task Duration(us)': [50, 5, 100]
        })
    
    def test_calculate_hbm_valid_inputs(self):
        test_cases = [
            (("INT32", "2,3"), [24 / 1024]), 
            (("FLOAT;INT32", '"1,2,3";4'), [12/1024, 16/1024]),
            (("INT32;FLOAT", "5,5;10"), [100/1024, 20/1024]), 
            (("FLOAT", "  100 , 200 "), [40000/1024]),
        ]
        for input_val, expected in test_cases:
            with self.subTest(input=input_val, expected=expected):
                result = self.analyzer.calculate_hbms(input_val[0], input_val[1])
                self.assertEqual(result, expected)

    def test_calculate_hbm_empty_values(self):
        test_cases = [
            (("", "1,2")),
            (("INT32", "")), 
            (("", "")),
        ]
        for case in test_cases:
            with self.subTest(input=case):
                result = self.analyzer.calculate_hbms(case[0], case[1])
                self.assertEqual(result, [])        

    def test_calculate_total_hbm_list_input(self):
        test_cases = [
            ([], 10, 10), 
            ([2, 3.5], 0, 5.5),
            ([-5, 10], 3, 8), 
            ([1, 2, 3], -2, 4)
        ]
        for op_hbm, start, expected in test_cases:
            with self.subTest(op_hbm=op_hbm, start=start):
                result = self.analyzer.calculate_total_hbms(op_hbm, start)
                self.assertAlmostEqual(result, expected)
    
    def test_init_given_valid_args_when_initializing_then_attributes_set_correctly(self):
        self.assertEqual(self.analyzer.fused_op_summary, "fused.csv")
        self.assertEqual(self.analyzer.origin_op_summary, "origin.csv")
        self.assertEqual(self.analyzer.output_path, "output_dir")
        self.assertEqual(self.analyzer.ge_graph_path, "ge_graph_path")
        self.assertEqual(self.analyzer.ops_mapping_json, os.path.join("output_dir", "ge_proto_build.json"))

    @patch('subprocess.run')
    def test_convert_ge_graph_given_valid_command_when_running_atc_then_conversion_success(self, mock_run):
        mock_run.return_value.returncode = 0
        self.analyzer.convert_ge_graph()
        mock_run.assert_called_once()
        self.assertTrue(mock_run.call_args[1]['stdout'] is subprocess.PIPE)

    @patch('subprocess.run')
    def test_convert_ge_graph_given_invalid_command_when_running_atc_then_conversion_fails(self, mock_run):
        mock_run.return_value.returncode = 1
        with self.assertRaises(RuntimeError):
            self.analyzer.convert_ge_graph()

    @patch('msit_prof.analyze.autofuse.single_op_analyze.get_all_subgraph')
    def test_get_fuse_graph_to_origin_op_mappig_given_valid_op_when_parsing_graph_then_mapping_populated(
        self, mock_get_all_subgraph
    ):
        mock_get_all_subgraph.return_value = [
            {
                'op': [
                    {
                        'type': 'AscBackend',
                        'name': 'fuse_op_1',
                        'attr': [
                            {'key': '_datadump_original_op_names', 'value': {'list': {'s': ['op1', 'op2']}}},
                            {'key': '_datadump_original_op_types', 'value': {'list': {'s': ['Type1', 'Type2']}}}
                        ]
                    }
                ]
            }
        ]
        analyzer = SingleOpAnalyzer(self.args)
        analyzer.get_fuse_graph_to_origin_op_mapping()
        self.assertIn('fuse_op_1', analyzer.ops_mapping_dict)
        self.assertEqual(len(analyzer.ops_mapping_dict['fuse_op_1']), 2)
        op_info = analyzer.ops_mapping_dict['fuse_op_1'][0]
        self.assertEqual(op_info.op_name, 'op1')
        self.assertEqual(op_info.op_type, 'Type1')

    @patch('pandas.read_csv')
    def test_load_op_summary_given_valid_csv_files_when_loading_then_dataframes_created(self, mock_read_csv):
        mock_read_csv.side_effect = [
            pd.DataFrame({'Op Name': ['op1'], 'OP Type': ['AscBackend'], 'Task Duration(us)': [100]}),
            pd.DataFrame({'Op Name': ['op1'], 'OP Type': ['Conv2D'], 'Task Duration(us)': [50]})
        ]
        can_compare_fuse_nodes = self.analyzer.load_op_summary()
        self.assertIsNotNone(self.analyzer.fuse_df)
        self.assertIsNotNone(self.analyzer.origin_df)
        self.assertEqual(len(can_compare_fuse_nodes), 1)

    def test_get_filter_origin_df_given_valid_fuse_node_when_filtering_then_origin_df_returned(self):
        self.analyzer.ops_mapping_dict = {'fuse_op': [OpInfo('op1')]}
        analyze_result = defaultdict(list)
        origin_op_df = self.analyzer.get_filter_origin_df('fuse_op', analyze_result)
        self.assertIsNotNone(origin_op_df)
        self.assertIn('op1', origin_op_df['Op Name'].values)

    def test_get_filter_origin_df_given_invalid_fuse_node_when_filtering_then_none_returned(self):
        self.analyzer.ops_mapping_dict = {}
        analyze_result = defaultdict(list)
        origin_op_df = self.analyzer.get_filter_origin_df('fuse_op', analyze_result)
        self.assertIsNone(origin_op_df)

    def test_analyze_origin_ops_given_valid_fuse_node_and_origin_ops_found_when_analyzing_then_origin_ops_returned(self):
        self.analyzer.ops_mapping_dict = {'fuse_op': [OpInfo('op1'), OpInfo('op2')]}
        total_origin_op_name = {'op1', 'op2', 'op3'}
        origin_op_duration_sum, origin_op_hbms, origin_op_hbms_sum, origin_op_hbms_save, not_found_op_list = self.analyzer.analyze_origin_ops('fuse_op', total_origin_op_name)
        self.assertEqual(len(origin_op_duration_sum), 2)
        self.assertEqual(len(not_found_op_list), 0)
        self.assertEqual(len(origin_op_hbms), 2)
        self.assertEqual(len(origin_op_hbms_sum), 1)
        self.assertEqual(len(origin_op_hbms_save), 2)

    def test_compute_performance_diff_given_valid_dataframes_when_computing_then_performance_diff_calculated(self):
        analyze_result = defaultdict(list)
        single_fused_df = pd.DataFrame(
            {'Task Duration(us)': [100], 
            "Op Name": 'fuse', 
            "OP Type": "my",
            'Input Data Types': 'FLOAT',
            'Input Shapes': 128,
            'Output Data Types': 'FLOAT',
            'Output Shapes': 121},
        )
        origin_op_df = pd.DataFrame(
            {'Task Duration(us)': [50], 
            "Op Name": 'ori', 
            "OP Type": "cpu",
            'Input Data Types': 'FLOAT',
            'Input Shapes': 512,
            'Output Data Types': 'FLOAT',
            'Output Shapes': 512}
        )
        origin_op_hbms_save = [1234, 5678]
        SingleOpAnalyzer.compute_performance_diff(single_fused_df, origin_op_df, analyze_result, 'fuse_op', origin_op_hbms_save)
        self.assertEqual(analyze_result["Time Ratio"][0], 2.0)
        self.assertEqual(analyze_result["Time Difference"][0], 50.0)
        self.assertEqual(analyze_result["HBMs Difference"][0], '(input:-1233.75, output:-5677.763671875)')
        self.assertEqual(analyze_result["HBMs Ratio"][0], '(input:0.0002025931928687196, output:4.162171979570271e-05)')
        self.assertEqual(analyze_result["Fused HBMs(KB)"][0], '(input:0.25, output:0.236328125)')

    @patch('pandas.DataFrame.to_csv')
    def test_save_analyze_result_given_valid_result_when_saving_then_csv_created(self, mock_to_csv):
        analyze_result = defaultdict(list)
        analyze_result["Fuse OpName"].append("fuse_op")
        analyze_result["Fuse OpType"].append("AscBackend")
        analyze_result["Fused Durations(us)"].append(100.0)
        analyze_result["Origin Durations(us)"].append(50.0)
        analyze_result["Time Ratio"].append(2.0)
        analyze_result["Time Difference"].append(50.0)
        analyze_result["Origin Duration(us) Each Op"].append("(op1, 50)")
        analyze_result["Not Found Origin Op"].append("")
        self.analyzer.save_analyze_result(analyze_result)
        mock_to_csv.assert_called_once()

    @patch.multiple('msit_prof.analyze.autofuse.single_op_analyze.SingleOpAnalyzer', 
                    convert_ge_graph=MagicMock(), get_fuse_graph_to_origin_op_mapping=MagicMock(),
                    load_op_summary=MagicMock(return_value={'fuse_op'}), get_filter_origin_df=MagicMock(),
                    compute_performance_diff=MagicMock(), save_analyze_result=MagicMock(),
                    analyze_origin_ops=MagicMock(return_value=([], [], [], [], [])))
    def test_analyze_given_valid_input_when_analyzing_then_analysis_complete(self):
        self.analyzer.fused_name_to_type = {'fuse_op': "unknown"}
        self.analyzer.analyze()
        self.analyzer.convert_ge_graph.assert_called_once()
        self.analyzer.get_fuse_graph_to_origin_op_mapping.assert_called_once()
        self.analyzer.load_op_summary.assert_called_once()
        self.analyzer.save_analyze_result.assert_called_once()