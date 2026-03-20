`timescale 1ns / 1ps

module tb_streamingRope_fsdb;

  string fsdb_path = "work/hdl/matlab_rope/streamingRope_hdl_entry/streamingRope_hdl_entry.fsdb";

  localparam int LANES = 8;
  localparam int NUM_TOKENS = 64;
  localparam int NUM_HEADS = 12;
  localparam int BEATS_PER_HEAD = 16;
  localparam int BEATS_PER_TOKEN = NUM_HEADS * BEATS_PER_HEAD;
  localparam int EXPECTED_BEATS = NUM_TOKENS * BEATS_PER_TOKEN;

  reg clk = 1'b0;
  reg reset = 1'b1;
  reg clk_enable = 1'b1;
  reg start = 1'b0;
  reg [15:0] cfgNumTokens = NUM_TOKENS;
  reg [7:0] cfgNumHeads = NUM_HEADS;
  reg [63:0] inBeat_0 = 64'h0;
  reg [63:0] inBeat_1 = 64'h0;
  reg [63:0] inBeat_2 = 64'h0;
  reg [63:0] inBeat_3 = 64'h0;
  reg [63:0] inBeat_4 = 64'h0;
  reg [63:0] inBeat_5 = 64'h0;
  reg [63:0] inBeat_6 = 64'h0;
  reg [63:0] inBeat_7 = 64'h0;
  reg inValid = 1'b0;

  wire ce_out;
  wire [63:0] outBeat_0;
  wire [63:0] outBeat_1;
  wire [63:0] outBeat_2;
  wire [63:0] outBeat_3;
  wire [63:0] outBeat_4;
  wire [63:0] outBeat_5;
  wire [63:0] outBeat_6;
  wire [63:0] outBeat_7;
  wire outValid;
  wire busy;
  wire done;

  integer sim_cycle_count = 0;
  integer busy_cycle_count = 0;
  integer out_valid_cycle_count = 0;
  integer out_count = 0;

  streamingRope_hdl_entry dut (
    .clk(clk),
    .reset(reset),
    .clk_enable(clk_enable),
    .start(start),
    .cfgNumTokens(cfgNumTokens),
    .cfgNumHeads(cfgNumHeads),
    .inBeat_0(inBeat_0),
    .inBeat_1(inBeat_1),
    .inBeat_2(inBeat_2),
    .inBeat_3(inBeat_3),
    .inBeat_4(inBeat_4),
    .inBeat_5(inBeat_5),
    .inBeat_6(inBeat_6),
    .inBeat_7(inBeat_7),
    .inValid(inValid),
    .ce_out(ce_out),
    .outBeat_0(outBeat_0),
    .outBeat_1(outBeat_1),
    .outBeat_2(outBeat_2),
    .outBeat_3(outBeat_3),
    .outBeat_4(outBeat_4),
    .outBeat_5(outBeat_5),
    .outBeat_6(outBeat_6),
    .outBeat_7(outBeat_7),
    .outValid(outValid),
    .busy(busy),
    .done(done)
  );

  initial begin
    $fsdbDumpfile(fsdb_path);
    $fsdbDumpvars(0, tb_streamingRope_fsdb, "+all");
  end

  always #5 clk = ~clk;

  function automatic [63:0] to_bits(input real value);
    to_bits = $realtobits(value);
  endfunction

  function automatic real lane_value(input int token_idx, input int head_idx, input int beat_idx, input int lane_idx);
    lane_value = -0.75 + token_idx * 0.03125 + head_idx * 0.0078125 + beat_idx * 0.001953125 + lane_idx * 0.000244140625;
  endfunction

  task automatic drive_beat(input int flat_beat_idx);
    integer token_idx;
    integer token_beat_idx;
    integer head_idx;
    integer beat_idx;
    begin
      token_idx = flat_beat_idx / BEATS_PER_TOKEN;
      token_beat_idx = flat_beat_idx % BEATS_PER_TOKEN;
      head_idx = token_beat_idx / BEATS_PER_HEAD;
      beat_idx = token_beat_idx % BEATS_PER_HEAD;

      inBeat_0 = to_bits(lane_value(token_idx, head_idx, beat_idx, 0));
      inBeat_1 = to_bits(lane_value(token_idx, head_idx, beat_idx, 1));
      inBeat_2 = to_bits(lane_value(token_idx, head_idx, beat_idx, 2));
      inBeat_3 = to_bits(lane_value(token_idx, head_idx, beat_idx, 3));
      inBeat_4 = to_bits(lane_value(token_idx, head_idx, beat_idx, 4));
      inBeat_5 = to_bits(lane_value(token_idx, head_idx, beat_idx, 5));
      inBeat_6 = to_bits(lane_value(token_idx, head_idx, beat_idx, 6));
      inBeat_7 = to_bits(lane_value(token_idx, head_idx, beat_idx, 7));
      inValid = 1'b1;
    end
  endtask

  task automatic clear_inputs;
    begin
      inBeat_0 = 64'h0;
      inBeat_1 = 64'h0;
      inBeat_2 = 64'h0;
      inBeat_3 = 64'h0;
      inBeat_4 = 64'h0;
      inBeat_5 = 64'h0;
      inBeat_6 = 64'h0;
      inBeat_7 = 64'h0;
      inValid = 1'b0;
    end
  endtask

  task automatic print_util_summary;
    begin
      $display("TB_DONE expected_beats=%0d observed_beats=%0d", EXPECTED_BEATS, out_count);
      $display("TB_UTIL cycles=%0d busy=%0d busy_pct=%0.3f out_valid=%0d out_valid_pct=%0.3f", sim_cycle_count, busy_cycle_count, (busy_cycle_count * 100.0) / sim_cycle_count, out_valid_cycle_count, (out_valid_cycle_count * 100.0) / sim_cycle_count);
    end
  endtask

  always @(posedge clk) begin
    if (reset) begin
      sim_cycle_count <= 0;
      busy_cycle_count <= 0;
      out_valid_cycle_count <= 0;
      out_count <= 0;
    end else begin
      sim_cycle_count <= sim_cycle_count + 1;
      if (busy) begin
        busy_cycle_count <= busy_cycle_count + 1;
      end
      if (outValid) begin
        out_valid_cycle_count <= out_valid_cycle_count + 1;
        out_count <= out_count + 1;
      end
    end
  end

  initial begin
    integer beat_idx;

    clear_inputs();
    repeat (4) @(negedge clk);
    reset = 1'b0;

    for (beat_idx = 0; beat_idx < EXPECTED_BEATS; beat_idx = beat_idx + 1) begin
      @(negedge clk);
      start = (beat_idx == 0);
      drive_beat(beat_idx);
    end

    @(negedge clk);
    start = 1'b0;
    clear_inputs();

    wait(done === 1'b1);
    @(posedge clk);
    #1;
    print_util_summary();

    if (out_count != EXPECTED_BEATS) begin
      $fatal(1, "Observed %0d output beats, expected %0d", out_count, EXPECTED_BEATS);
    end

    #20;
    $finish;
  end

endmodule