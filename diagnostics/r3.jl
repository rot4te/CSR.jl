using CSR, GTPSA
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
geom0 = CSRGeometry([0.2, 0.3, 0.2], [0.0, 1.5, 0.0], c3)
function t(name, f)
  print(rpad(name,40))
  try; f(); println("OK"); catch e; println("FAIL ", first(split(sprint(showerror,e),'\n'))); 
    ENV["BT"]==name && (Base.showerror(stdout,e,catch_backtrace()); println()); end
end
ENV["BT"] = "static-after-d2"
d1 = Descriptor(2,2)
xs = @vars(d1)
xd = @vars(d1, dynamic=true)
t("static-before-d2", () -> CSRKickPoint(geom0, 2, 0.17, 200.0 + xs[1]))
t("dynamic-before-d2", () -> CSRKickPoint(geom0, 2, 0.17, 200.0 + xd[1]))
d2 = Descriptor(3,2)
t("static-after-d2", () -> CSRKickPoint(geom0, 2, 0.17, 200.0 + xs[1]))
t("dynamic-after-d2", () -> CSRKickPoint(geom0, 2, 0.17, 200.0 + xd[1]))
