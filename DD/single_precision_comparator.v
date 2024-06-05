`timescale 1ns / 1ps

module single_precision_comparator(
    input [31:0] A_32,
    input [31:0] B_32,
    output reg equal_to,
    output reg less_than,
    output reg greater_than
);

    always @* begin
        // Initialize the results to '0'
        equal_to = 0;
        less_than = 0;
        greater_than = 0;

        // Special Cases Handling
        if (A_32[30:23] == 255 || B_32[30:23] == 255) begin
            equal_to = 1;
            less_than = 1;
            greater_than = 1;
        end else if (A_32[30:23] == 0 || B_32[30:23] == 0) begin
            equal_to = 0;
            less_than = 0;
            greater_than = 0;
        end else if (A_32[31] == B_32[31] && A_32[30:23] == B_32[30:23] && A_32[22:0] == B_32[22:0]) begin
            equal_to = 1;
            less_than = 0;
            greater_than = 0;
        end else begin
            // Sign comparison
            if (A_32[31] != B_32[31]) begin
                greater_than = (A_32[31] == 0) && (B_32[31] == 1);
                less_than = (A_32[31] == 1) && (B_32[31] == 0);
            end else begin
                // Exponent comparison
                if (A_32[30:23] > B_32[30:23]) begin
                    greater_than = (A_32[31] == 0) ? 1 : 0;
                    less_than = (A_32[31] == 0) ? 0 : 1;
                end else if (A_32[30:23] < B_32[30:23]) begin
                    greater_than = (A_32[31] == 0) ? 0 : 1;
                    less_than = (A_32[31] == 0) ? 1 : 0;
                end else begin
                    // Mantissa comparison
                    if (A_32[22:0] > B_32[22:0]) begin
                        greater_than = (A_32[31] == 0) ? 1 : 0;
                        less_than = (A_32[31] == 0) ? 0 : 1;
                    end else if (A_32[22:0] < B_32[22:0]) begin
                        greater_than = (A_32[31] == 0) ? 0 : 1;
                        less_than = (A_32[31] == 0) ? 1 : 0;
                    end
                end
            end
        end
    end
endmodule
