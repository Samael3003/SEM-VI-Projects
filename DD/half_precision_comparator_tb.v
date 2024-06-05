module half_precision_comparator_tb;

    reg [15:0] A_16;
    reg [15:0] B_16;
    wire equal_to;
    wire less_than;
    wire greater_than;

    real A;
    real B;

    half_precision_comparator uut (
        .A_16(A_16),
        .B_16(B_16),
        .equal_to(equal_to),
        .less_than(less_than),
        .greater_than(greater_than)
    );

    // Provide input values and monitor results
    initial begin
        // Test Case 1: Equal numbers (Positives)
        A = 5.4;
        B = 5.4;
        A_16 = $realtobits(A);
        B_16 = $realtobits(B);
        $display("Test Case 1:");
        $display("Number1: %h", A_16);
        $display("Number2: %h", B_16);
        $display("Equal: %b", equal_to);
        $display("Less Than: %b", less_than);
        $display("Greater Than: %b", greater_than);
        #10;

        // Test Case 2: Equal numbers (Negatives)
        A = -5.4;
        B = -5.4;
        A_16 = $realtobits(A);
        B_16 = $realtobits(B);
        $display("Test Case 2:");
        $display("Number1: %h", A_16);
        $display("Number2: %h", B_16);
        $display("Equal: %b", equal_to);
        $display("Less Than: %b", less_than);
        $display("Greater Than: %b", greater_than);
        #10;

        // Test Case 3: A > B (Positives)
        A = 7.2;
        B = 6.3;
        A_16 = $realtobits(A);
        B_16 = $realtobits(B);
        $display("Test Case 3:");
        $display("Number1: %h", A_16);
        $display("Number2: %h", B_16);
        $display("Equal: %b", equal_to);
        $display("Less Than: %b", less_than);
        $display("Greater Than: %b", greater_than);
        #10;

        // Test Case 4: A > B (Negatives)
        A = -6.3;
        B = -7.2;
        A_16 = $realtobits(A);
        B_16 = $realtobits(B);
        $display("Test Case 4:");
        $display("Number1: %h", A_16);
        $display("Number2: %h", B_16);
        $display("Equal: %b", equal_to);
        $display("Less Than: %b", less_than);
        $display("Greater Than: %b", greater_than);
        #10;

        // Test Case 5: A < B (Positives)
        A = 8.1;
        B = 9.0;
        A_16 = $realtobits(A);
        B_16 = $realtobits(B);
        $display("Test Case 5:");
        $display("Number1: %h", A_16);
        $display("Number2: %h", B_16);
        $display("Equal: %b", equal_to);
        $display("Less Than: %b", less_than);
        $display("Greater Than: %b", greater_than);
        #10;

        // Test Case 6: A < B (Negatives)
        A = -9.0;
        B = -8.1;
        A_16 = $realtobits(A);
        B_16 = $realtobits(B);
        $display("Test Case 6:");
        $display("Number1: %h", A_16);
        $display("Number2: %h", B_16);
        $display("Equal: %b", equal_to);
        $display("Less Than: %b", less_than);
        $display("Greater Than: %b", greater_than);
        #10;

        // Test Case 7: Zeros
        A = 0.0;
        B = 0.0;
        A_16 = $realtobits(A);
        B_16 = $realtobits(B);
        $display("Test Case 7:");
        $display("Number1: %h", A_16);
        $display("Number2: %h", B_16);
        $display("Equal: %b", equal_to);
        $display("Less Than: %b", less_than);
        $display("Greater Than: %b", greater_than);
        #10;

        // Test Case 8: Infinity
        A = 0/0;
        B = 0/0;
        A_16 = $realtobits(A);
        B_16 = $realtobits(B);
        $display("Test Case 8:");
        $display("Number1: %h", A_16);
        $display("Number2: %h", B_16);
        $display("Equal: %b", equal_to);
        $display("Less Than: %b", less_than);
        $display("Greater Than: %b", greater_than);
        #10;

        // Test Case 9: NaN (Not a Number)
        A = 16'h7FF80000;
        B = 1;
        A_16 = A;
        B_16 = $realtobits(B);
        $display("Test Case 9:");
        $display("Number1: %h", A_16);
        $display("Number2: %h", B_16);
        $display("Equal: %b", equal_to);
        $display("Less Than: %b", less_than);
        $display("Greater Than: %b", greater_than);
        #10;

        // Finish simulation
        $finish;
    end
endmodule
