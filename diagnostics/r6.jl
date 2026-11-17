using CSR, GTPSA
c3 = zeros(4, 6); c3[:, 1] = [0.0, 4e-4, 1.1e-3, 2e-3]; c3[:, 2] = [2e-3, 1.5e-3, 3e-3, 3e-3]; c3[:, 6] = [0.0, 1e-3, -2e-3, 0.0]
geom0 = CSRGeometry([0.2, 0.3, 0.2], [0.0, 1.5, 0.0], c3)
d = Descriptor(2,2); x = @vars(d)
d2 = Descriptor(3,2); y = @vars(d2)
function t(name, f)
  print(rpad(name,40))
  try; f(); println("OK"); catch e; println("FAIL ", first(split(sprint(showerror,e),'\n'))); end
end
t("two statics mixed s_kick/gamma", () -> CSRKickPoint(geom0, 2, 0.17 + x[1], 200.0 + y[1]))
t("static + dynamic", () -> CSRKickPoint(geom0, 2, 0.17 + x[1], 200.0 + (@vars(d2, dynamic=true))[1]))
t("promote_type", () -> promote_type(typeof(x[1]), typeof(y[1])))
t("x + y", () -> x[1] + y[1])
t("Float64(x)", () -> Float64(x[1]))
t("convert", () -> convert(Float64, x[1]))
t("zeros(1) .= x", () -> (a = zeros(1); a[1] = x[1]))
t("promote_type(static,Float64)", () -> (@show promote_type(typeof(x[1]), Float64)))
t("promote_type(static,Int)", () -> (@show promote_type(typeof(x[1]), Int)))
t("float(promote_type 4)", () -> (@show float(promote_type(Float64, typeof(x[1]), Float64, typeof(x[1])))))
