// =============================================================
// th_lif_top.v
// -------------------------------------------------------------
// Arty A7-35T board wrapper for TH-LIF single neuron.
//
// Behavior after reset release:
//   - step_idx starts at 0
//   - Tick fires immediately, then after every `done` pulse
//   - Stimulus ROM provides I_in indexed by step_idx
//   - step_idx auto-wraps 128 → 0 (infinite loop)
//
// Visible output (human-eye speed):
//   - LD4 (H5):  spike flash, extended to 100 ms per spike event
//   - LD5 (J5):  done pulse (visible as faint glow during fast operation)
//   - LD6 (T9):  step_idx[5]  (toggles every 32 steps → phase boundaries)
//   - LD7 (T10): step_idx[6]  (toggles every 64 steps → first/second half)
//
// ILA probes (via mark_debug):
//   - dbg_tick, dbg_done, dbg_spike
//   - dbg_I_in[15:0], dbg_V_out[15:0], dbg_step[6:0]
// =============================================================
`timescale 1ns / 1ps

module th_lif_top (
    input  wire        CLK100MHZ,   // 100 MHz on E3
    input  wire        ck_rst,       // CPU_RESETN active-low button (C2)
    output wire  [3:0] led           // LD4..LD7
);

    // --- Clock/reset aliases ---
    wire clk   = CLK100MHZ;
    wire rst_n = ck_rst;

    // --- Forward declarations (with mark_debug for ILA) ---
    (* mark_debug = "true" *) wire done;
    (* mark_debug = "true" *) wire spike;
    (* mark_debug = "true" *) wire signed [15:0] V_out;
    (* mark_debug = "true" *) wire signed [15:0] I_in;

    // --- Tick generator: pulse after each done, bootstrap on reset release ---
    (* mark_debug = "true" *) reg tick;
    reg started;
    always @(posedge clk) begin
        if (!rst_n) begin
            tick    <= 1'b0;
            started <= 1'b0;
        end else if (!started) begin
            tick    <= 1'b1;
            started <= 1'b1;
        end else if (done) begin
            tick    <= 1'b1;
        end else begin
            tick    <= 1'b0;
        end
    end

    // --- Step counter (auto-wraps 128 → 0 via 7-bit overflow) ---
    (* mark_debug = "true" *) reg [6:0] step_idx;
    always @(posedge clk) begin
        if (!rst_n)      step_idx <= 7'd0;
        else if (done)   step_idx <= step_idx + 1'b1;
    end

    // --- Stimulus ROM ---
    wire [15:0] stim_data;
    stimulus_rom u_stim (
        .addr (step_idx),
        .data (stim_data)
    );
    assign I_in = $signed(stim_data);

    // --- TH-LIF neuron (verified in Step 2) ---
    th_lif_neuron u_neuron (
        .clk   (clk),
        .rst_n (rst_n),
        .tick  (tick),
        .I_in  (I_in),
        .done  (done),
        .spike (spike),
        .V_out (V_out)
    );

    // --- Spike LED extender: 100 ms monostable ---
    // spike is 1-cycle wide; we latch it and hold LED high for HOLD_CYCLES
    localparam integer HOLD_CYCLES = 10_000_000;   // 100 ms @ 100 MHz
    reg [23:0] led_timer;
    reg        spike_led;
    always @(posedge clk) begin
        if (!rst_n) begin
            led_timer <= 24'd0;
            spike_led <= 1'b0;
        end else if (spike) begin
            led_timer <= HOLD_CYCLES[23:0];
            spike_led <= 1'b1;
        end else if (led_timer != 24'd0) begin
            led_timer <= led_timer - 1'b1;
            spike_led <= 1'b1;
        end else begin
            spike_led <= 1'b0;
        end
    end

    // --- LED routing ---
    assign led[0] = spike_led;        // LD4: visible spike
    assign led[1] = done;             // LD5: step tick (very brief)
    assign led[2] = step_idx[5];      // LD6: toggles at each phase boundary
    assign led[3] = step_idx[6];      // LD7: toggles at half-sequence

endmodule
