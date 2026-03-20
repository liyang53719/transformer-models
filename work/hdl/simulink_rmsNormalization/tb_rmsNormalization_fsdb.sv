`timescale 1ns / 1ps

module tb_rmsNormalization_fsdb;

  string fsdb_path = "work/hdl/simulink_rmsNormalization/rmsNormalization/rmsNormalization.fsdb";

  localparam int LANES = 8;
  localparam int HIDDEN_SIZE = 1536;
  localparam int NUM_TOKENS = 64;
  localparam int BEATS_PER_TOKEN = HIDDEN_SIZE / LANES;
  localparam int EXPECTED_BEATS = NUM_TOKENS * BEATS_PER_TOKEN;
  localparam real EPSILON = 1536e-6;
  localparam real COMPARE_ABS_TOL = 1.0e-4;
  localparam real COMPARE_REL_TOL = 1.0e-6;

  reg clk = 1'b0;
  reg reset = 1'b1;
  reg clk_enable = 1'b1;
  reg start = 1'b0;
  reg [31:0] cfgGammaBeat_lane [0:LANES-1];
  reg [255:0] cfgGammaBeat;
  reg cfgGammaValid = 1'b0;
  reg [31:0] ddrDataBeat_lane [0:LANES-1];
  reg [255:0] ddrDataBeat;
  reg ddrDataValid = 1'b0;

  wire ce_out;
  wire [255:0] outBeat;
  wire outValid;
  wire [15:0] ddrReadAddr;
  wire ddrReadEn;
  wire done;
  wire busy;

  integer beat_count = 0;
  integer out_count = 0;
  integer x_valid_count = 0;
  integer inv_rms_valid_count = 0;
  integer compare_beat_count = 0;
  integer compare_lane_count = 0;
  integer compare_fail_count = 0;
  integer compare_token_idx = 0;
  integer compare_token_beat_idx = 0;
  integer sum_compare_token_idx = 0;
  integer inv_compare_token_idx = 0;
  integer pending_addr = 0;
  reg pending_valid = 1'b0;
  real max_abs_err = 0.0;
  real max_rel_err = 0.0;
  shortreal token_scale [0:NUM_TOKENS-1];
  real token_sum_sq [0:NUM_TOKENS-1];
  real token_inv_rms [0:NUM_TOKENS-1];

  DUTPacked dut (
    .clk(clk),
    .reset(reset),
    .clk_enable(clk_enable),
    .start(start),
    .cfgGammaBeat(cfgGammaBeat),
    .cfgGammaValid(cfgGammaValid),
    .ddrDataBeat(ddrDataBeat),
    .ddrDataValid(ddrDataValid),
    .ce_out(ce_out),
    .outBeat(outBeat),
    .outValid(outValid),
    .ddrReadAddr(ddrReadAddr),
    .ddrReadEn(ddrReadEn),
    .done(done),
    .busy(busy)
  );

  initial begin
    $fsdbDumpfile(fsdb_path);
    $fsdbDumpvars(0, tb_rmsNormalization_fsdb, "+all");
  end

  always #5 clk = ~clk;

  function automatic [31:0] to_bits(input shortreal value);
    to_bits = $shortrealtobits(value);
  endfunction

  function automatic shortreal gamma_value(input int beat_idx, input int lane_idx);
    gamma_value = shortreal'(0.25 + beat_idx * 0.001 + lane_idx * 0.0001);
  endfunction

  function automatic shortreal x_value(input int token_idx, input int beat_idx, input int lane_idx);
    x_value = shortreal'(-0.75 + token_idx * 0.01 + beat_idx * 0.001 + lane_idx * 0.0001);
  endfunction

  function automatic [31:0] gamma_word(input int beat_idx, input int lane_idx);
    begin
      gamma_word = to_bits(gamma_value(beat_idx, lane_idx));
    end
  endfunction

  function automatic [31:0] x_word(input int token_idx, input int beat_idx, input int lane_idx);
    begin
      x_word = to_bits(x_value(token_idx, beat_idx, lane_idx));
    end
  endfunction

  function automatic real abs_real(input real value);
    begin
      if (value < 0.0) begin
        abs_real = -value;
      end else begin
        abs_real = value;
      end
    end
  endfunction

  function automatic [31:0] packed_lane_word(input [255:0] packed_beat, input int lane_idx);
    packed_lane_word = packed_beat[(lane_idx * 32) +: 32];
  endfunction

  function automatic shortreal expected_value(input int token_idx, input int beat_idx, input int lane_idx);
    expected_value = shortreal'(x_value(token_idx, beat_idx, lane_idx) * gamma_value(beat_idx, lane_idx) * token_scale[token_idx]);
  endfunction

  task automatic check_output_beat(input [255:0] packed_beat);
    integer lane_idx;
    real observed_value;
    real expected_value_real;
    real abs_err;
    real rel_err;
    real rel_den;
    begin
      for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
        observed_value = $bitstoshortreal(packed_lane_word(packed_beat, lane_idx));
        expected_value_real = expected_value(compare_token_idx, compare_token_beat_idx, lane_idx);
        abs_err = abs_real(observed_value - expected_value_real);
        rel_den = abs_real(expected_value_real);
        if (rel_den < 1.0e-12) begin
          rel_den = 1.0e-12;
        end
        rel_err = abs_err / rel_den;

        if (abs_err > max_abs_err) begin
          max_abs_err = abs_err;
        end
        if (rel_err > max_rel_err) begin
          max_rel_err = rel_err;
        end

        if (abs_err > COMPARE_ABS_TOL && rel_err > COMPARE_REL_TOL) begin
          compare_fail_count = compare_fail_count + 1;
          $display("TB_COMPARE_FAIL token=%0d beat=%0d lane=%0d observed=%0.9g expected=%0.9g abs_err=%0.9g rel_err=%0.9g", compare_token_idx, compare_token_beat_idx, lane_idx, observed_value, expected_value_real, abs_err, rel_err);
        end
      end

      compare_beat_count = compare_beat_count + 1;
      compare_lane_count = compare_lane_count + LANES;

      if (compare_token_beat_idx == BEATS_PER_TOKEN - 1) begin
        compare_token_beat_idx = 0;
        compare_token_idx = compare_token_idx + 1;
      end else begin
        compare_token_beat_idx = compare_token_beat_idx + 1;
      end
    end
  endtask

  task automatic drive_gamma_beat(input int beat_idx);
    integer lane_idx;
    begin
      for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
        cfgGammaBeat_lane[lane_idx] = gamma_word(beat_idx, lane_idx);
      end
      cfgGammaBeat = {cfgGammaBeat_lane[7], cfgGammaBeat_lane[6], cfgGammaBeat_lane[5], cfgGammaBeat_lane[4], cfgGammaBeat_lane[3], cfgGammaBeat_lane[2], cfgGammaBeat_lane[1], cfgGammaBeat_lane[0]};
      cfgGammaValid = 1'b1;
      @(negedge clk);
      cfgGammaValid = 1'b0;
      for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
        cfgGammaBeat_lane[lane_idx] = 32'h00000000;
      end
      cfgGammaBeat = 256'h0;
    end
  endtask

  task automatic clear_ddr_bus;
    integer lane_idx;
    begin
      ddrDataValid = 1'b0;
      for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
        ddrDataBeat_lane[lane_idx] = 32'h00000000;
      end
      ddrDataBeat = 256'h0;
    end
  endtask

  always @* begin
    integer token_idx;
    integer beat_idx;
    integer lane_idx;

    if (pending_valid) begin
      token_idx = pending_addr / BEATS_PER_TOKEN;
      beat_idx = pending_addr % BEATS_PER_TOKEN;
      ddrDataValid = 1'b1;
      for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
        ddrDataBeat_lane[lane_idx] = x_word(token_idx, beat_idx, lane_idx);
      end
      ddrDataBeat = {ddrDataBeat_lane[7], ddrDataBeat_lane[6], ddrDataBeat_lane[5], ddrDataBeat_lane[4], ddrDataBeat_lane[3], ddrDataBeat_lane[2], ddrDataBeat_lane[1], ddrDataBeat_lane[0]};
    end else begin
      clear_ddr_bus();
    end
  end

  always @(posedge clk) begin
    if (reset) begin
      pending_valid <= 1'b0;
      pending_addr <= 0;
      beat_count <= 0;
      out_count <= 0;
    end else begin
      if (pending_valid) begin
        beat_count <= beat_count + 1;
      end

      pending_valid <= ddrReadEn;
      pending_addr <= ddrReadAddr;

      if (outValid) begin
        check_output_beat(outBeat);
        out_count <= out_count + 1;
      end

      if (dut.u_CoreDUT.TokenSram_readValid) begin
        x_valid_count <= x_valid_count + 1;
      end

      if (dut.u_CoreDUT.captureInvRms) begin
        if (sum_compare_token_idx < 4) begin
          $display(
            "TB_SUM_DBG token=%0d beats_seen=%0d current_sum_valid=%0d observed=%0.9g expected=%0.9g ratio=%0.9g",
            sum_compare_token_idx,
            beat_count,
            dut.u_CoreDUT.BeatAccumulator_currentSumValid,
            $bitstoshortreal(dut.u_CoreDUT.BeatAccumulator_currentSum),
            token_sum_sq[sum_compare_token_idx],
            $bitstoshortreal(dut.u_CoreDUT.BeatAccumulator_currentSum) / token_sum_sq[sum_compare_token_idx]
          );
        end
        sum_compare_token_idx <= sum_compare_token_idx + 1;
      end

      if (dut.u_CoreDUT.InvRmsLatch_invRmsLatchedValid) begin
        inv_rms_valid_count <= inv_rms_valid_count + 1;
        if (inv_compare_token_idx < 4) begin
          $display(
            "TB_INVRMS_DBG token=%0d observed=%0.9g expected=%0.9g ratio=%0.9g",
            inv_compare_token_idx,
            $bitstoshortreal(dut.u_CoreDUT.InvRmsLatch_invRmsLatched),
            token_inv_rms[inv_compare_token_idx],
            $bitstoshortreal(dut.u_CoreDUT.InvRmsLatch_invRmsLatched) / token_inv_rms[inv_compare_token_idx]
          );
        end
        inv_compare_token_idx <= inv_compare_token_idx + 1;
      end
    end
  end

  initial begin
    integer beat_idx;
    integer lane_idx;
    integer token_idx;
    real sum_sq;

    for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
      cfgGammaBeat_lane[lane_idx] = 32'h00000000;
      ddrDataBeat_lane[lane_idx] = 32'h00000000;
    end
    for (token_idx = 0; token_idx < NUM_TOKENS; token_idx = token_idx + 1) begin
      sum_sq = 0.0;
      for (beat_idx = 0; beat_idx < BEATS_PER_TOKEN; beat_idx = beat_idx + 1) begin
        for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
          sum_sq = sum_sq + x_value(token_idx, beat_idx, lane_idx) * x_value(token_idx, beat_idx, lane_idx);
        end
      end
      token_sum_sq[token_idx] = sum_sq;
      token_inv_rms[token_idx] = 1.0 / $sqrt(sum_sq + EPSILON);
      token_scale[token_idx] = shortreal'($sqrt(HIDDEN_SIZE * 1.0) / $sqrt(sum_sq + EPSILON));
    end
    cfgGammaBeat = 256'h0;
    ddrDataBeat = 256'h0;

    repeat (5) @(negedge clk);
    reset <= 1'b0;

    for (beat_idx = 0; beat_idx < BEATS_PER_TOKEN; beat_idx = beat_idx + 1) begin
      drive_gamma_beat(beat_idx);
    end

    @(negedge clk);
    start <= 1'b1;
    @(negedge clk);
    start <= 1'b0;
  end

  initial begin
    fork
      begin
        integer drain_cycles;
        wait(done === 1'b1);
        drain_cycles = 0;
        while (out_count < EXPECTED_BEATS && drain_cycles < 128) begin
          @(posedge clk);
          drain_cycles = drain_cycles + 1;
        end
        if (compare_beat_count != EXPECTED_BEATS) begin
          $fatal(1, "TB compare beat count mismatch: expected=%0d observed=%0d", EXPECTED_BEATS, compare_beat_count);
        end
        if (compare_fail_count != 0) begin
          $fatal(1, "TB numeric compare failed: compare_fail_count=%0d max_abs_err=%0.9g max_rel_err=%0.9g", compare_fail_count, max_abs_err, max_rel_err);
        end
        $display("TB_COMPARE_OK beats_compared=%0d lanes_compared=%0d max_abs_err=%0.9g max_rel_err=%0.9g", compare_beat_count, compare_lane_count, max_abs_err, max_rel_err);
        $display("TB_DONE beats_requested=%0d x_valid_count=%0d beats_produced=%0d inv_rms_valid_count=%0d compare_beats=%0d compare_lanes=%0d max_abs_err=%0.9g max_rel_err=%0.9g", beat_count, x_valid_count, out_count, inv_rms_valid_count, compare_beat_count, compare_lane_count, max_abs_err, max_rel_err);
        $finish;
      end
      begin
        repeat (30000) @(posedge clk);
        $display("TB_TIMEOUT beats_requested=%0d x_valid_count=%0d beats_produced=%0d inv_rms_valid_count=%0d compare_beats=%0d compare_lanes=%0d max_abs_err=%0.9g max_rel_err=%0.9g pending_valid=%0d pending_addr=%0d ddrReadEn=%0d ddrReadAddr=%0d busy=%0d done=%0d", beat_count, x_valid_count, out_count, inv_rms_valid_count, compare_beat_count, compare_lane_count, max_abs_err, max_rel_err, pending_valid, pending_addr, ddrReadEn, ddrReadAddr, busy, done);
        $fatal(1, "TB timeout waiting for done");
      end
    join_any
    disable fork;
  end

endmodule