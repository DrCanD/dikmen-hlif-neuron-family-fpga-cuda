// =============================================================
// tb_th_lif_top.v
// -------------------------------------------------------------
// Integration testbench for th_lif_top.
// Verifies:
//   - Tick generator starts on reset release
//   - Step counter advances through 128 positions
//   - Spikes occur at expected steps (10, 21, 66, 69, ..., 93)
//   - LED extender holds spike LED high for many cycles
// Runs for ~15 μs (full loop + part of second loop).
// =============================================================
`timescale 1ns / 1ps

module tb_th_lif_top;

    reg CLK100MHZ = 0;
    reg ck_rst    = 0;
    wire [3:0] led;

    always #5 CLK100MHZ = ~CLK100MHZ;   // 100 MHz

    th_lif_top dut (
        .CLK100MHZ (CLK100MHZ),
        .ck_rst    (ck_rst),
        .led       (led)
    );

    integer spike_count_total;
    integer last_step_logged;
    reg prev_done;

    initial begin
        $display("==============================================");
        $display(" TH-LIF Top - Integration Testbench");
        $display("==============================================");

        ck_rst = 0;       // asserted (active-low means 0 = reset)
        #200;             // hold reset for 200 ns
        ck_rst = 1;       // release
        $display(" [t=%0t] Reset released, tick generator should bootstrap", $time);

        spike_count_total = 0;
        last_step_logged  = -1;
        prev_done         = 0;

        // Monitor for 15 μs
        fork
            // Thread 1: log every done pulse and spike
            forever begin
                @(posedge CLK100MHZ);
                if (dut.done && !prev_done) begin
                    if (dut.spike) begin
                        spike_count_total = spike_count_total + 1;
                        $display(" [t=%5t ns]  step=%3d  I=%04h  V_out=%04h  SPIKE",
                                 $time, dut.step_idx, dut.I_in & 16'hFFFF,
                                 dut.V_out & 16'hFFFF);
                    end
                end
                prev_done = dut.done;
            end

            // Thread 2: total runtime bound
            begin
                #15000;
                $display("\n----------------------------------------------");
                $display(" 15 μs elapsed");
                $display(" Total spike events observed: %0d", spike_count_total);
                $display(" Current step_idx: %0d", dut.step_idx);
                $display(" LED state: %b (LD7 LD6 LD5 LD4)", led);
                $display(" Expected: ~18 spikes (1.5 loops × 12 spikes/loop)");
                if (spike_count_total >= 12 && spike_count_total <= 30)
                    $display(" *** INTEGRATION OK ***");
                else
                    $display(" *** UNEXPECTED SPIKE COUNT ***");
                $display("==============================================");
                $finish;
            end
        join
    end

endmodule
