`timescale 1ns / 1ps

module tb_rmsNormalization_fsdb;

  localparam int LANES = 8;
  localparam int HIDDEN_SIZE = 1536;
  localparam int NUM_TOKENS = 64;
  localparam int BEATS_PER_TOKEN = HIDDEN_SIZE / LANES;

  reg clk = 1'b0;
  reg reset = 1'b1;
  reg clk_enable = 1'b1;
  reg reset_1 = 1'b0;
  reg start = 1'b0;
  reg [31:0] cfgGammaBeat [0:LANES-1];
  reg cfgGammaValid = 1'b0;
  reg [31:0] ddrDataBeat [0:LANES-1];
  reg ddrDataValid = 1'b0;

  wire ce_out;
  wire [31:0] outBeat [0:LANES-1];
  wire outValid;
  wire [15:0] ddrReadAddr;
  wire ddrReadEn;
  wire done;
  wire busy;

  integer beat_count = 0;
  integer out_count = 0;
  integer pending_addr = 0;
  reg pending_valid = 1'b0;

  DUT dut (
    .clk(clk),
    .reset(reset),
    .clk_enable(clk_enable),
    .reset_1(reset_1),
    .start(start),
    .cfgGammaBeat_0(cfgGammaBeat[0]),
    .cfgGammaBeat_1(cfgGammaBeat[1]),
    .cfgGammaBeat_2(cfgGammaBeat[2]),
    .cfgGammaBeat_3(cfgGammaBeat[3]),
    .cfgGammaBeat_4(cfgGammaBeat[4]),
    .cfgGammaBeat_5(cfgGammaBeat[5]),
    .cfgGammaBeat_6(cfgGammaBeat[6]),
    .cfgGammaBeat_7(cfgGammaBeat[7]),
    .cfgGammaValid(cfgGammaValid),
    .ddrDataBeat_0(ddrDataBeat[0]),
    .ddrDataBeat_1(ddrDataBeat[1]),
    .ddrDataBeat_2(ddrDataBeat[2]),
    .ddrDataBeat_3(ddrDataBeat[3]),
    .ddrDataBeat_4(ddrDataBeat[4]),
    .ddrDataBeat_5(ddrDataBeat[5]),
    .ddrDataBeat_6(ddrDataBeat[6]),
    .ddrDataBeat_7(ddrDataBeat[7]),
    .ddrDataValid(ddrDataValid),
    .ce_out(ce_out),
    .outBeat_0(outBeat[0]),
    .outBeat_1(outBeat[1]),
    .outBeat_2(outBeat[2]),
    .outBeat_3(outBeat[3]),
    .outBeat_4(outBeat[4]),
    .outBeat_5(outBeat[5]),
    .outBeat_6(outBeat[6]),
    .outBeat_7(outBeat[7]),
    .outValid(outValid),
    .ddrReadAddr(ddrReadAddr),
    .ddrReadEn(ddrReadEn),
    .done(done),
    .busy(busy)
  );

  always #5 clk = ~clk;

  function automatic [31:0] to_bits(input shortreal value);
    to_bits = $shortrealtobits(value);
  endfunction

  function automatic [31:0] gamma_word(input int beat_idx, input int lane_idx);
    shortreal value;
    begin
      value = shortreal'(0.25 + beat_idx * 0.001 + lane_idx * 0.0001);
      gamma_word = to_bits(value);
    end
  endfunction

  function automatic [31:0] x_word(input int token_idx, input int beat_idx, input int lane_idx);
    shortreal value;
    begin
      value = shortreal'(-0.75 + token_idx * 0.01 + beat_idx * 0.001 + lane_idx * 0.0001);
      x_word = to_bits(value);
    end
  endfunction

  task automatic drive_gamma_beat(input int beat_idx);
    integer lane_idx;
    begin
      for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
        cfgGammaBeat[lane_idx] = gamma_word(beat_idx, lane_idx);
      end
      cfgGammaValid = 1'b1;
      @(negedge clk);
      cfgGammaValid = 1'b0;
      for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
        cfgGammaBeat[lane_idx] = 32'h00000000;
      end
    end
  endtask

  task automatic clear_ddr_bus;
    integer lane_idx;
    begin
      ddrDataValid = 1'b0;
      for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
        ddrDataBeat[lane_idx] = 32'h00000000;
      end
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
        ddrDataBeat[lane_idx] = x_word(token_idx, beat_idx, lane_idx);
      end
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
        out_count <= out_count + 1;
      end
    end
  end

  initial begin
    integer beat_idx;
    integer lane_idx;

    $fsdbDumpfile("work/hdl/simulink_rmsNormalization/rmsNormalization/rmsNormalization.fsdb");
    $fsdbDumpvars(0, tb_rmsNormalization_fsdb, "+all");

    for (lane_idx = 0; lane_idx < LANES; lane_idx = lane_idx + 1) begin
      cfgGammaBeat[lane_idx] = 32'h00000000;
      ddrDataBeat[lane_idx] = 32'h00000000;
    end

    repeat (5) @(negedge clk);
    reset <= 1'b0;

    reset_1 <= 1'b1;
    @(negedge clk);
    reset_1 <= 1'b0;

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
        wait(done === 1'b1);
        repeat (10) @(posedge clk);
        $display("TB_DONE beats_requested=%0d beats_produced=%0d", beat_count, out_count);
        $finish;
      end
      begin
        repeat (200000) @(posedge clk);
        $fatal(1, "TB timeout waiting for done");
      end
    join_any
    disable fork;
  end

endmodule