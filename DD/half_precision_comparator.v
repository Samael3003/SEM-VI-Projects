`timescale 1ns / 1ps

module half_precision_comparator(
    input [15:0] A_16,
    input [15:0] B_16,
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
        if (A_16[14:10] == 31 || B_16[14:10] == 31) begin
            equal_to = 1;
            less_than = 1;
            greater_than = 1;
        end else if (A_16[14:10] == 0 || B_16[14:10] == 0) begin
            equal_to = 0;
            less_than = 0;
            greater_than = 0;
        end else if (A_16[15] == B_16[15] && A_16[14:10] == B_16[14:10] && A_16[9:0] == B_16[9:0]) begin
            equal_to = 1;
            less_than = 0;
            greater_than = 0;
        end else begin
            // Sign comparison
            if (A_16[15] != B_16[15]) begin
                greater_than = (A_16[15] == 0) && (B_16[15] == 1);
                less_than = (A_16[15] == 1) && (B_16[15] == 0);
            end else begin
                // Exponent comparison
                if (A_16[14:10] > B_16[14:10]) begin
                    greater_than = (A_16[15] == 0) ? 1 : 0;
                    less_than = (A_16[15] == 0) ? 0 : 1;
                end else if (A_16[14:10] < B_16[14:10]) begin
                    greater_than = (A_16[15] == 0) ? 0 : 1;
                    less_than = (A_16[15] == 0) ? 1 : 0;
                end else begin
                    // Mantissa comparison
                    if (A_16[9:0] > B_16[9:0]) begin
                        greater_than = (A_16[15] == 0) ? 1 : 0;
                        less_than = (A_16[15] == 0) ? 0 : 1;
                    end else if (A_16[9:0] < B_16[9:0]) begin
                        greater_than = (A_16[15] == 0) ? 0 : 1;
                        less_than = (A_16[15] == 0) ? 1 : 0;
                    end
                end
            end
        end
    end
endmodule
